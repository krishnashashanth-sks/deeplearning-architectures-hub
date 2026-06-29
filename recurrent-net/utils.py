import numpy as np

def generate_time_series(num_points):
  time=np.linspace(0,100,num_points)
  data=np.sin(time/10)+np.random.normal(scale=0.1,size=num_points)
  return data.reshape(-1,1)

def create_sequences(data,look_back):
  X,y=[],[]
  for i in range(len(data)-look_back):
    X.append(data[i:(i+look_back),0])
    y.append(data[i+look_back,0])
  return np.array(X),np.array(y)