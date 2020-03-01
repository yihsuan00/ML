#### HW2 MLfoundation

##### introduce function

###### Getdata
- variable: `size`, `ratio`(default = 0.2)
- to make the data with the noise ratio 0.2
- record the data sign

###### Theta
- variable: `data`, `size`(default = 20)
- to consider the two ends point and the adjacent points

###### Decision_Stump 
- variable: `data`, `label`, `size`
- consider the case of `s`= 1 and `s` = -1
- calculate error and return the best_s, best_theta and errot

###### calculate
- variable" `times`, `figname`, `size`
- compute for the `times` iterations
- append the Ein and Eout into array 
- plot the histogram of `Ein-Eout`
- save the fig with `figname`

##### how to operate the code
- Problem 7
    - python ML_HW2.py 1000 -u Hw_2.7.1 20
- Problem 8
    - python ML_HW2.py 1000 -u Hw_2.8.1 2000

