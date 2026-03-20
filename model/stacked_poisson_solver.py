import os
import typing

import torch

from .solver import Solver
import util

class StackedPoissonSolver(Solver):
    """
    Biharmonic solver via operator splitting:
        ∇⁴u = f  →  ∇²v = f,  ∇²u = v
    
    Two Solver instances are chained: the first solves the intermediate
    Poisson problem for v, the second solves for u using v as the RHS.

    Both sub-solvers are forced to biharmonic_problem=False (they solve
    Poisson, not biharmonic), and the parent __init__ is called with
    biharmonic_problem=True so that residual checks in eval mode use
    the correct biharmonic residual for the outer convergence criterion.

    Boundary conditions for the intermediate variable v:
        The BCs for v (i.e. ∇²u on the boundary) are generally not known
        in closed form. We default to v=0 on the boundary, which is exact
        for clamped-plate problems (where ∂u/∂n = 0 on ∂Ω).
    """

    def __init__(self,
                 # --- architecture args (shared by both sub-solvers) ---
                 structure: str,
                 downsampling_policy: str,
                 upsampling_policy: str,
                 device: torch.device,
                 num_iterations: int,
                 relative_tolerance: float,
                 initialize_x0: str,
                 num_mg_layers: int,
                 num_mg_pre_smoothing: int,
                 num_mg_post_smoothing: int,
                 activation: str,
                 initialize_trainable_parameters: str,
                 # --- splitting-specific ---
                 num_iterations_v: typing.Optional[int] = None,
                 relative_tolerance_v: typing.Optional[float] = None):

        # Parent stores meta-attributes and sets up residual checking with
        # biharmonic=True.  We skip creating a UGrid iterator here by
        # temporarily patching structure to a no-op, then overwriting.
        super().__init__(
            structure=structure,
            downsampling_policy=downsampling_policy,
            upsampling_policy=upsampling_policy,
            device=device,
            num_iterations=num_iterations,
            relative_tolerance=relative_tolerance,
            initialize_x0=initialize_x0,
            num_mg_layers=num_mg_layers,
            num_mg_pre_smoothing=num_mg_pre_smoothing,
            num_mg_post_smoothing=num_mg_post_smoothing,
            activation=activation,
            initialize_trainable_parameters=initialize_trainable_parameters,
            biharmonic_smoother=True,   # outer residual uses biharmonic check
        )

        # Allow independent iteration budgets for each sub-solve.
        # If not specified, mirror the outer solver's settings.
        num_iter_v = num_iterations_v if num_iterations_v is not None else num_iterations
        rel_tol_v  = relative_tolerance_v if relative_tolerance_v is not None else relative_tolerance

        shared_kwargs = dict(
            structure=structure,
            downsampling_policy=downsampling_policy,
            upsampling_policy=upsampling_policy,
            device=device,
            initialize_x0=initialize_x0,
            num_mg_layers=num_mg_layers,
            num_mg_pre_smoothing=num_mg_pre_smoothing,
            num_mg_post_smoothing=num_mg_post_smoothing,
            activation=activation,
            initialize_trainable_parameters=initialize_trainable_parameters,
            biharmonic_smoother=False,   # both sub-solvers are Poisson
        )

        # Sub-solver 1: ∇²v = f  (intermediate Laplacian solve)
        self.solver_v: Solver = Solver(
            num_iterations=num_iter_v,
            relative_tolerance=rel_tol_v,
            **shared_kwargs,
        )

        # Sub-solver 2: ∇²u = v  (final Laplacian solve)
        self.solver_u: Solver = Solver(
            num_iterations=num_iterations,
            relative_tolerance=relative_tolerance,
            **shared_kwargs,
        )

    def _intermediate_bc(
        self,
        bc_value: torch.Tensor,
        bc_mask: torch.Tensor,
    ) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (bc_value_v, bc_mask_v) for the intermediate solve ∇²v = f.

        Default: v = 0 on all boundary pixels (clamped-plate assumption).
        Override this method for other physical BC types, e.g.:
          - Simply supported plate: v = ∇²u_exact on boundary (known)
          - Free edge: more involved, requires a mixed formulation
        """
        bc_value_v = torch.zeros_like(bc_value)
        bc_mask_v  = bc_mask                       # same mask, value pinned to 0
        return bc_value_v, bc_mask_v
    
    def __call__(
        self,
        x: typing.Optional[torch.Tensor],
        bc_value: torch.Tensor,
        bc_mask: torch.Tensor,
        f: typing.Optional[torch.Tensor],
        rel_tol: typing.Optional[float] = None,
    ) -> typing.Tuple[torch.Tensor, int]:
        """
        Solves the biharmonic ∇⁴u = f as two chained Poisson solves.

        Step 1: solve  ∇²v = f  with v=0 on ∂Ω  →  v
        Step 2: solve  ∇²u = v  with u=bc_value on ∂Ω  →  u

        Returns (u, total_iterations).
        """
        if not self.is_train:
            if rel_tol is None:
                rel_tol = self.relative_tolerance

        # ── Step 1: intermediate Poisson solve for v ──────────────────────────
        bc_value_v, bc_mask_v = self._intermediate_bc(bc_value, bc_mask)

        # f is the RHS of the biharmonic; it becomes the RHS of ∇²v = f.
        # If f is None, the biharmonic is ∇⁴u = 0, so ∇²v = 0 → v = 0
        # everywhere, and step 2 reduces to a plain Laplace solve for u.
        v, iters_v = self.solver_v(
            x=None,              # always start v from scratch
            bc_value=bc_value_v,
            bc_mask=bc_mask_v,
            f=f,
            rel_tol=rel_tol,
        )

        # ── Step 2: Poisson solve for u using v as RHS ────────────────────────
        # v plays the role of f in ∇²u = v
        u, iters_u = self.solver_u(
            x=x,                 # warm-start u from caller's initial guess
            bc_value=bc_value,
            bc_mask=bc_mask,
            f=v,                 # <-- the output of step 1 is the RHS here
            rel_tol=rel_tol,
        )

        return u, iters_v + iters_u

    def train(self):
        self.is_train = True
        self.solver_v.train()
        self.solver_u.train()

    def eval(self):
        self.is_train = False
        self.solver_v.eval()
        self.solver_u.eval()

    def parameters(self):
        """Yield parameters from both sub-solvers for a single optimiser."""
        yield from self.solver_v.parameters()
        yield from self.solver_u.parameters()

    def save(self, checkpoint_path: str, epoch: int):
        self.solver_v.save(os.path.join(checkpoint_path, 'solver_v'), epoch)
        self.solver_u.save(os.path.join(checkpoint_path, 'solver_u'), epoch)

    def load(self, checkpoint_path: str, epoch: int):
        path_v = self.solver_v.load(os.path.join(checkpoint_path, 'solver_v'), epoch)
        path_u = self.solver_u.load(os.path.join(checkpoint_path, 'solver_u'), epoch)
        return path_v, path_u