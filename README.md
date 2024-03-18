# Keyboard Simulated Annealing Layout Optimizer

Supporting simulated annealing code for the [Why I Made The World's Worst Keyboard](https://youtu.be/188fipF-i5I) YouTube video.

To train on your own custom dataset and customize keyboard layout or algorithm arguments,
copy all your training data (text files) to the folder `persistent/raw_data`,
copy `data-commented.json` to `persistent/data.json`, remove comments and adjust parameters.

## Adjusting penalty weights

The weights are in `persistent/data.json` and `data-commented.json` has some comments explaining how to edit it. Some hints for adjusting them:

Note that by also adding effortBiases together with effortWeights, you actually have an efforts neuron that can be used to track back (fit) the parameters that get the closest to a specific keyboard layout.

- Distance: **DO NOT** put too much weight on distance or distance growth penalties or the algorithm will tend to put all the important keys clustered together in the middle;
  - yScale scales down movements across y axis so that lateral movements (across x axis) will deal more penalty, this reduces lateral movements, it can have some undesirable results for non-ortholinear keyboard layouts;
- Single hand: If you prefer to use a single hand for most words instead of favoring using both, put negative weight in single hand penalty;
- Double finger prevents using same finger twice, will favor putting keys that are not normally written together in the same finger columns;
- FingersCPS and rowsCPS (clicks per second) penalties favor your strongest fingers and the strongest rows of the keyboard;
  - RowsCPSBias biases specific rows, even if your CPS in that specific row isn't the best, I used it to remove a little bit the home row and increase the bias of the row below it;
- Left hand reward favors using left hand, let it 0 if you don't want it or negative to favor right hand;

## Running

Download at <https://julialang.org/downloads/> and install the Julia language.

Assuming `julia` is in your path:

For CUDA GPU, run:

```bash
git clone https://github.com/gabrielmaia2/sa-keyboard-layout-optimizer.git
cd sa-keyboard-layout-optimizer
julia --project -L "revise.jl"
main(useGPU=true, findWorst=false)
```

For threaded CPU, change <nthreads> for the number of threads your processor can run and run:

```bash
git clone https://github.com/gabrielmaia2/sa-keyboard-layout-optimizer.git
cd sa-keyboard-layout-optimizer
julia -t<nthreads> --project -L "revise.jl"
main(useGPU=false, findWorst=false)
```

To find the worst keyboard layout instead of the best, just change `findWorst=false` to `findWorst=true`.
