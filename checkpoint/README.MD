biharmonic_1
- This uses the update: y = 0.05 * f - y in kernel.py

biharmonic_2
- 3 smoothing, 6 mg layers
- uses         y = y + 0.05 * f
- this will be kept for future iterations (results are pretty much the same as before)
- 

biharmonic_3 (4 mg, 5 smoothing)
- changed mg_layers from 6 to 4, smoothing from 3 to 5
- changed weighting from 0.2 to 0.35
- smaller batch size
- uses MSE loss
- best results so far

biharmonic_4 (4 mg, 6 smoothing)
- pretty bad output
- increased smoothing to 6
- max epoch is 225
- data used 0.1 to 0.15

biharmonic_5 (4 mg, 6 smoothing)
- still pretty bad output, omega 0.35 or 0.25 doesn't matter
- scheduler "step" "50" "0.1" -> scheduler "step" "20" "0.5"

biharomnic_6 (5 mg, 5 smoothing)
- back to 5 smoothing layers, omega 0.35, will now try 5 multigrid layers
- results better than biharmonic 5, but worse than biharmonic_3

biharmonic_7 (6 mg, 5 smoothing)
- will do 5 smoothing layers, but multigrid layers from 5 -> 6, data to 0.1
- output also kinda bad

biharmonic_8 (4mg, 5 smoothing)
- mirroring setup in biharmonic_3, but now with the revised scheduler
- output is actually pretty poor

biharmonic_9 (4mg, 5 smoothing)
- Bringing back the old scheduler, and increased max epoch from 100 > 150