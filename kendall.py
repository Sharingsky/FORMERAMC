from pandas import DataFrame
import pandas as pd
x = [10-i for i in range(10)]
y = [8,2,9,3,5,10,1,4,7,6]
data = DataFrame({'x':x,'y':y})
print(data.head())
kend=data.corr(method='kendall')
print(kend)