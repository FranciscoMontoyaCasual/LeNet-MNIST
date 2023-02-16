# LeNet

LeNet es una arquitectura de CNN, la cual esta compuesta de 7 capas. La red neuronal esta compuesta de 3 capas convolucionales, 2 capas de subsampling y 2 capas fully connected. La descripcion de cada una de las capas es la siguiente:

- Conv2d 1: in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=0
- Max-Pooling: kernel_size=2
- Conv2d 2: in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0
- Max-Pooling: kernel_size=2
- Conv2d 3: in_channels=16, out_channels=120, kernel_size=5, stride=1, padding=0
- Max-Pooling: kernel_size=2
- Linear 1: in_features=120, out_features=84
- Linear 2: in_features=84, out_features=10

Con esta arquitectura y con los parametros presentes de entrnamiento se consigue una presición de entrenamiento y validación del 99% en menos de 15 epocas. 
