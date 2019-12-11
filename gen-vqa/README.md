# Gen-VQA

## Usage

#### 1. Download and Preproccess VQA Datasets

```bash
cd utils
./preprocess.sh
# make sure to chmod 775 all scripts
```

#### 2. Open a terminal

```bash
visdom
```

#### 3. Open Another Terminal and Train

```bash
python3 runtime.py --batch_size=64 --lr=0.0002
```

#### 4. To Visualize Progress

```bash
ssh  -L 9999:localhost:8097 username@ip.of.remote.server
```

Open Chrome and go to [`localhost:9999`](localhost:9999)

#### 5. A Shortcut for Visualization

Open Chrome and go to [`ip.of.remote.server:8097`](ip.of.remote.server:8097)

For example `abcd.csail.mit.edu:8097`

