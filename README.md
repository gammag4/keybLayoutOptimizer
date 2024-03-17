# Keyboard Simulated Annealing Layout Optimizer

Supporting simulated annealing code for the [Why I Made The World's Worst Keyboard](https://youtu.be/188fipF-i5I) YouTube video.

To train on your own custom dataset and customize keyboard layout or algorithm arguments,
copy all your training data (text files) to the folder `persistent/raw_data`,
copy `data-commented.json` to `persistent/data.json`, remove comments and adjust parameters.

## Running

Download at <https://julialang.org/downloads/> and install the Julia language.

Assuming `julia` is in your path:

For CUDA GPU, run:

```bash
git clone https://github.com/gabrielmaia2/sa-keyboard-layout-optimizer.git
cd sa-keyboard-layout-optimizer
julia --project=. -L main.jl
main(useGPU=true)
```

For threaded CPU, change nthreads for the number of threads your processor can run and run:

```bash
git clone https://github.com/gabrielmaia2/sa-keyboard-layout-optimizer.git
cd sa-keyboard-layout-optimizer
julia -t<nthreads> --project=. -L main.jl
main(useGPU=false)
```
