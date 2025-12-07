import argparse
import shutil
import time
import typing
import os

from loguru import logger
import numpy as np
import torch
import torch.backends.cudnn
import torch.utils.data

from arg import TestArg
from data import SynDat
from model import Solver
import util


# noinspection DuplicatedCode
def test_on_dataset(model, test_loader, device) -> None:
    with torch.no_grad():
        test_loss_dict: typing.Dict[str, typing.List[torch.Tensor]] = {}

        for batch in test_loader:
            x: typing.Optional[torch.Tensor] = None
            bc_value: torch.Tensor = batch['bc_value'].to(device)
            bc_mask: torch.Tensor = batch['bc_mask'].to(device)
            f: typing.Optional[torch.Tensor] = None

            tup: typing.Tuple[torch.Tensor, np.int64] = model(x, bc_value, bc_mask, f)
            y, iterations_used = tup

            absolute_loss, relative_loss = util.relative_residue(y, bc_value, bc_mask, f)
            absolute_loss = absolute_loss.mean()
            relative_loss = relative_loss.mean()

            iterations_used = torch.tensor([iterations_used], dtype=torch.float32).to(device)

            if 'absolute_loss' in test_loss_dict:
                test_loss_dict['absolute_loss'].append(absolute_loss)
            else:
                test_loss_dict['absolute_loss']: typing.List[torch.Tensor] = [absolute_loss]

            if 'relative_loss' in test_loss_dict:
                test_loss_dict['relative_loss'].append(relative_loss)
            else:
                test_loss_dict['relative_loss']: typing.List[torch.Tensor] = [relative_loss]

            if 'iterations_used' in test_loss_dict:
                test_loss_dict['iterations_used'].append(iterations_used)
            else:
                test_loss_dict['iterations_used']: typing.List[torch.Tensor] = [iterations_used]

        for k, v in test_loss_dict.items():
            logger.info('[Test] {} = {}'.format(k, torch.mean(torch.tensor(v))))
def test_on_single_data_color(
        img_path: str,
        model: Solver,
        device: torch.device,
        benchmark_iteration: typing.Optional[int] = None
    ) -> None:

    import cv2
    import time

    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Image not found: {img_path}")

    H, W, C = img.shape
    assert C == 3, "Input image must be 3-channel color"
    B, G, R = cv2.split(img)

    # Boundary: non-zero pixel means boundary
    bc_mask_np = ((R > 0) | (G > 0) | (B > 0)).astype(np.float32)

    def make_tensors(channel_np):
        bc_value_np = channel_np.astype(np.float32)

        f_np = channel_np.astype(np.float32)
        f_np[bc_mask_np == 1] = 0

        bc_mask = torch.from_numpy(bc_mask_np).unsqueeze(0).unsqueeze(0).to(device)
        bc_value = torch.from_numpy(bc_value_np).unsqueeze(0).unsqueeze(0).to(device)
        f = torch.from_numpy(f_np).unsqueeze(0).unsqueeze(0).to(device)
        return bc_mask, bc_value, f, bc_mask_np, bc_value_np, f_np

    R_bc_mask, R_bc_value, R_f, R_mask_np, R_bc_np, R_f_np = make_tensors(R)
    G_bc_mask, G_bc_value, G_f, G_mask_np, G_bc_np, G_f_np = make_tensors(G)
    B_bc_mask, B_bc_value, B_f, B_mask_np, B_bc_np, B_f_np = make_tensors(B)

    test_loss_dict: typing.Dict[str, typing.List[torch.Tensor]] = {}
    time_lst: typing.List[float] = []

    if benchmark_iteration is None:
        benchmark_iteration = 1

    def solve_and_measure(bc_mask, bc_value, f):
        """Run one channel through the model and compute residuals."""
        with torch.no_grad():
            y, iterations_used = model(None, bc_value, bc_mask, f)

            abs_residual_norm, rel_residual_norm = util.relative_residue(
                y, bc_value, bc_mask, f
            )

            abs_residual_norm = abs_residual_norm.mean()
            rel_residual_norm = rel_residual_norm.mean()

            return y, iterations_used, abs_residual_norm, rel_residual_norm

    # -------------------------------------------------
    # Run benchmark iterations
    # -------------------------------------------------
    for _ in range(benchmark_iteration):
        start = time.perf_counter_ns()

        # --- Solve R,G,B independently ---
        R_y, R_iter, R_abs, R_rel = solve_and_measure(R_bc_mask, R_bc_value, R_f)
        G_y, G_iter, G_abs, G_rel = solve_and_measure(G_bc_mask, G_bc_value, G_f)
        B_y, B_iter, B_abs, B_rel = solve_and_measure(B_bc_mask, B_bc_value, B_f)

        time_lst.append(time.perf_counter_ns() - start)

        # --- accumulate loss dict ---
        for k, v in [
            ("abs_residual_norm", (R_abs + G_abs + B_abs) / 3.0),
            ("rel_residual_norm", (R_rel + G_rel + B_rel) / 3.0),
            ("iterations_used", torch.tensor([(R_iter + G_iter + B_iter) / 3],
                                             dtype=torch.float32).to(device))
        ]:
            if k not in test_loss_dict:
                test_loss_dict[k] = []
            test_loss_dict[k].append(v)

    # -------------------------------------------------
    # Prepare outputs for visualization
    # -------------------------------------------------
    R_y_np = R_y.detach().cpu().squeeze().numpy()
    G_y_np = G_y.detach().cpu().squeeze().numpy()
    B_y_np = B_y.detach().cpu().squeeze().numpy()

    bc_mask_np = R_mask_np  # same for all 3 channels

    y_np = np.stack([R_y_np, G_y_np, B_y_np], axis=-1)
    y_np = np.clip(y_np, 0, 255).astype(np.uint8)

    avg_time_ms = np.mean(time_lst) / 1e6
    avg_rel = torch.mean(torch.stack(test_loss_dict["rel_residual_norm"])).item()

    log_str = f"ColorPoisson {avg_time_ms:.3f} ms, rel res {avg_rel:.4e}"
    logger.info(log_str)

    util.plt_subplot(
        dic={
            'mask': bc_mask_np,
            'f_R': R_f_np,
            'f_G': G_f_np,
            'f_B': B_f_np,
            'b_R': R_bc_np,
            'b_G': G_bc_np,
            'b_B': B_bc_np,
            'y': y_np
        },
        suptitle=log_str,
        show=False,
        dump=f"var/cmp_viz/{log_str}.png"
    )

    return

# noinspection DuplicatedCode
def test_on_single_data(testcase: str,
                        size: int,
                        model: Solver,
                        device: torch.device,
                        benchmark_iteration: typing.Optional[int] = None) \
        -> None:
    with torch.no_grad():
        test_loss_dict: typing.Dict[str, typing.List[torch.Tensor]] = {}

        image_size: int = size
        bc_value, bc_mask, f = util.get_testcase(testcase, image_size, device)

        time_lst: typing.List[float] = []

        if benchmark_iteration is None:
            benchmark_iteration = 1

        for _ in range(benchmark_iteration):
            start_time: float = time.perf_counter_ns()
            tup: typing.Tuple[torch.Tensor, np.int64] = model(None, bc_value, bc_mask, f)
            y, iterations_used = tup
            time_lst.append(time.perf_counter_ns() - start_time)

        # logger.info(f'[Test] Testcase {testcase} of size {size} min = {torch.min(y)} max = {torch.max(y)}')
        # np.save(f'var/testcase/npy/{testcase}_{size}.npy', np.array(time_lst))

        tup: typing.Tuple[torch.Tensor, torch.Tensor] = util.relative_residue(y, bc_value, bc_mask, f)
        abs_residual_norm, rel_residual_norm = tup
        abs_residual_norm: torch.Tensor = abs_residual_norm.mean()
        rel_residual_norm: torch.Tensor = rel_residual_norm.mean()

        iterations_used = torch.tensor([iterations_used], dtype=torch.float32).to(device)

        if 'abs_residual_norm' in test_loss_dict:
            test_loss_dict['abs_residual_norm'].append(abs_residual_norm)
        else:
            test_loss_dict['abs_residual_norm']: typing.List[torch.Tensor] = [abs_residual_norm]

        if 'rel_residual_norm' in test_loss_dict:
            test_loss_dict['rel_residual_norm'].append(rel_residual_norm)
        else:
            test_loss_dict['rel_residual_norm']: typing.List[torch.Tensor] = [rel_residual_norm]

        if 'iterations_used' in test_loss_dict:
            test_loss_dict['iterations_used'].append(iterations_used)
        else:
            test_loss_dict['iterations_used']: typing.List[torch.Tensor] = [iterations_used]

        # for k, v in test_loss_dict.items():
        #     logger.info('[Test] {} = {}'.format(k, torch.mean(torch.tensor(v))))

        bc_value_np: np.ndarray = bc_value.cpu().squeeze().numpy()
        bc_mask_np: np.ndarray = bc_mask.cpu().squeeze().numpy()
        f_np: typing.Optional[np.ndarray] = f.cpu().squeeze().numpy() if f is not None else None
        y_np: np.ndarray = y.cpu().squeeze().numpy()

        log_str = (f'UGrid {testcase}_{size} {np.mean(time_lst) / 1e6} ms, ' +
                   'rel res {}'.format(torch.mean(torch.tensor(test_loss_dict['rel_residual_norm'])).item()))
        logger.info(log_str)
        util.plt_subplot(
                dic={'m': bc_mask_np, 'f': f_np, 'b': bc_value_np, 'y': y_np},
                suptitle=log_str,
                show=False,
                dump=f'var/cmp_viz/{log_str}.png'
        )


# noinspection DuplicatedCode
def main() -> None:
    # argument parameters
    arg_opt: argparse.Namespace = TestArg().parse()

    # training parameters
    experiment_checkpoint_path: str = os.path.join(arg_opt.checkpoint_root, arg_opt.load_experiment)
    exp_opt_np: np.ndarray = np.load(os.path.join(experiment_checkpoint_path, 'opt.npy'), allow_pickle=True)

    # merged argument namespace
    opt: argparse.Namespace = util.merge_namespace(exp_opt_np.item(), arg_opt)

    logger.info('======================== Args ========================')
    for k, v in vars(opt).items():
        logger.info(f'{k}\t\t{v}')
    logger.info('======================================================\n')

    # backend
    device: torch.device = util.get_device()
    logger.info(f'[Test] Using device {device}')

    if opt.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        logger.info(f'[Test] Enforce deterministic algorithms, cudnn benchmark disabled')
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        logger.info(f'[Test] Do not enforce deterministic algorithms, cudnn benchmark enabled')

    if opt.seed is not None:
        torch.manual_seed(opt.seed)
        torch.cuda.manual_seed_all(opt.seed)
        logger.info(f'[Test] Manual seed PyTorch with seed {opt.seed}\n')
    else:
        seed: int = torch.seed()
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        logger.info(f'[Test] Using random seed {seed} for PyTorch\n')

    # model
    opt.num_iterations = 64
    model = Solver(opt.structure, opt.downsampling_policy, opt.upsampling_policy, device,
                   opt.num_iterations, 1e-4, opt.initialize_x0,
                   opt.num_mg_layers, opt.num_mg_pre_smoothing, opt.num_mg_post_smoothing,
                   opt.activation, 'default', 
                   jacobi_step_fn=util.biharmonic_jacobi_step if opt.jacobi_step_fn == "biharmonic" else util.jacobi_step)
    loaded_checkpoint = model.load(experiment_checkpoint_path, opt.load_epoch)
    model.eval()
    logger.info(f'[Test] Checkpoint loaded from {loaded_checkpoint}\n')

    # # test on dataset
    # # ##
    # test_dataset_path: str = os.path.join(opt.dataset_root, 'test')
    # test_dataset = SynDat(os.path.join(opt.dataset_root, 'test'))
    # test_loader = torch.utils.data.DataLoader(test_dataset,
    #                                           batch_size=opt.batch_size,
    #                                           num_workers=opt.num_workers,
    #                                           pin_memory=True)
    # logger.info(f'[Test] {len(test_dataset)} testing data loaded from {test_dataset_path}')
    # test_on_dataset(model, test_loader, device)

    testcase_lst: typing.List[str] = ['bag', 'cat', 'lock', 'note', 'poisson_region', 'punched_curve',
                                      'shape_l', 'shape_square', 'shape_square_poisson', 'star', 'bag']

    # Test UGrid
    # TODO: benchmark_iteration should be 100 for benchmarking.
    if opt.color_testcase_path is not None:
        test_on_single_data_color(opt.color_testcase_path, model, device, benchmark_iteration=10)
    else:
        for size in [1025, 257]:
            for testcase in testcase_lst:
                # os.mkdir('var/conv/UGrid/tmp/')
                test_on_single_data(testcase, size, model, device, benchmark_iteration=10)
                # shutil.move('var/conv/UGrid/tmp/', f'var/conv/UGrid/{testcase}_{size}/')


if __name__ == '__main__':
    main()








