# print("hello github")
import numpy as np

each_data=[[2,3,4,5]]
test_array=np.array([[1,2,3,4]])

print(test_array.shape)
each_data=np.concatenate((each_data,test_array),axis=1)
print(each_data)
each_data = np.expand_dims(each_data, axis=2)
print(each_data)
