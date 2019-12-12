# load dataset
from numpy import dstack
from pandas import read_csv

# carregar um único arquivo como uma array numpy
def load_file(filepath):
	dataframe = read_csv(filepath, header=None, delim_whitespace=True)
	return dataframe.values

# carrega uma lista de arquivos, como dados x, y, z, para uma determinada variável
def load_group(filenames, prefix=''):
	loaded = list()
	for name in filenames:
		data = load_file(prefix + name)
		loaded.append(data)
	# grupo de pilhas para que os recursos sejam da 3ª dimensão
	loaded = dstack(loaded)
	return loaded

# carrega um grupo de dataset, os de treinamento e os de teste
def load_dataset(group, prefix=''):
	filepath = prefix + group + '/Inertial Signals/'
	# carrega todos os  9 arquivos como um unico array
	filenames = list()
	# aceleração total
	filenames += ['total_acc_x_'+group+'.txt', 'total_acc_y_'+group+'.txt', 'total_acc_z_'+group+'.txt']
	# aceleração do corpo
	filenames += ['body_acc_x_'+group+'.txt', 'body_acc_y_'+group+'.txt', 'body_acc_z_'+group+'.txt']
	# giroscópio do corpo
	filenames += ['body_gyro_x_'+group+'.txt', 'body_gyro_y_'+group+'.txt', 'body_gyro_z_'+group+'.txt']
	# load input data
	X = load_group(filenames, filepath)
	# load class output
	y = load_file(prefix + group + '/y_'+group+'.txt')
	return X, y

# load all train
trainX, trainy = load_dataset('train', 'HARDataset/')
print(trainX.shape, trainy.shape)
# load all test
testX, testy = load_dataset('test', 'HARDataset/')
print(testX.shape, testy.shape) #Amostras, timesteps, recursos | 