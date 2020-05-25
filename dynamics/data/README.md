### Generating Data

Use the following script to generate data. Currently supports generation only through random policy. 
```
python dynamics/data/generate_master.py --root-dir datasets/carracing --policy random --episodes 1000 --threads 5
```

Random actions can be sampled based on two noise types such as *brown* and *white*. *Brown* seems to provide consistent
rollouts.