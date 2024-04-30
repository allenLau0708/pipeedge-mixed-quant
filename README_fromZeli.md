### Run the code from zero

#### Fork (and star) my repo, and then clone the forked to your local.
#### If you miss some package when running the code, just install it in a regular way(pip or conda)

#### Run the save_model_weights.py, If you have already download the model, you need to delete and re-download it.(We make some modification in this version)
##### make sure you at least successfully download ViT-B_16-224.npz,ViT-L_16-224.npz and resnet18.pt

#### download the dataset from https://www.image-net.org/challenges/LSVRC/index.php,ImageNet,Large Scale Visual Recognition Challenge 2012 (ILSVRC2012) (validation dataset, 6.3GB)
##### unzip and run the valprep.sh to the downloaded file.
##### If you run on discovery, you just need to modify evaluation.py from 189th line to 203th line, you don't need to do 2.and 2.1

#### Create a folder called 'result' in your project, if it is existing, make sure it is empty by deleting all inside files

#### make sure your project looks like:

project:
    evaluation.py
    runtime.py
    ViT-B_16-224.npz
    ViT-L_16-224.npz
    resnet18.pt
    ILSVRC2012_img_val:
        n0......:
            (lots of).JPEG
        n0......
        (lots of)
        n0.......
    pipeedge:
        others
        quantization:
            basic_op.py
            others
    result

#### control the clamp or noClamp (we can't automatically control it right now)
##### modify runtime.py from 98th line to 101th line
##### modify evaluation.py from 37th line to 38th line

#### If you run on discovery, modify the jobinfo
##### modify evaluation_tools/upload_eval.job
cd to your own path on discovery
choose the corresponding command
##### modify evaluation_tools/evaluation_partition.sh, choose the corresponding command

### The example of running on local:
Float-Quant at the 22th layer using 8 bit totally (6 bits for exponential) for resnet18 model.
Please wait at the beginning of the running, since the program will prepossess the dataset and the model, after a while, you will see the result line by line.

```sh
python evaluation.py -pt 1,10,11,21 -q 8,8 -e 6 -m torchvision/resnet18
```

### The example of running on discovery:
You could see the discovery generating thousands of jobs
They are all combinations of (bit,e,layer)
After while,you could see some results poping up at result/{model_name}/{job_name}.txt,If you could see the txt file, basically it means the program is running successfully

You could also see the output at evaluation_tools/slurm_{job_id}.out

```sh
cd evaluation_tools
./evaluation_partition.sh
```
Note: you may need to give evaluation_partition.sh a premium system-level permission at first.

### some useful command line

##### If you wanna see your job status
```sh
squeue --me
```

##### If you wanna see the counting of your running/waiting jobs
```sh
squeue --me | awk '
BEGIN {
    abbrev["R"]="(Running)"
    abbrev["PD"]="(Pending)"
    abbrev["CG"]="(Completing)"
    abbrev["F"]="(Failed)"
}
NR>1 {a[$5]++}
END {
    for (i in a) {
        printf "%-2s %-12s %d\n", i, abbrev[i], a[i]
    }
}'
```

