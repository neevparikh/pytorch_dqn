# Pytorch DQN Implementation
Implementation of DQN (Mnih 2015) in Pytorch.

Please create a virtualenv using the following method:

``` python3 -m venv env ```

Then, activate the venv using the following command:

``` source env/bin/activate ``` 

This repo uses Python 3.7 but should be fine on any recent Python 3 version.

Now, install minimal packages using: 

``` pip install -r requirements.txt ```

Make sure run.sh is executable (check with `ls -l`, if +x is in the permissions,
you're good). If not use `chmod +x run.sh`, to get it executable. 

Then use `./run.sh` to run the dqn on the specified environment.
