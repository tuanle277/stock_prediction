def getHParams(name=None):
	# Set up what's the same for each experiment
	hParams = {
		'experimentName': name,
		'epochs': 50,
	    'test_prop': 0.1,
	    'valid_prop':0.2,
	    'look_back': 20
	}
	shortTest = 0 # hardcode to 1 to run a quick debugging test
	if shortTest:
		print("+++++++++++++++++ WARNING: SHORT TEST +++++++++++++++++")
		hParams['datasetProportion'] = 0.0001
		hParams['epochs'] = 2

	if (name is None):
		# Not running an experiment yet, so just return the "common" parameters
		return hParams

	if (name == 'LSTM_128_64_Dense_32_1'):
		hParams['LSTMLayers'] = [
		{
			'LSTM_numLayers': 128, 
			'return_sequences': 1,
			'LSTM_act': 'relu', 
		},
		{
			'LSTM_numLayers': 64, 
			'return_sequences': 0,
			'LSTM_act': 'relu', 
		},
	]
		hParams['denseLayers'] = [32, 1]
		hParams['optimizer'] = 'adam'

	elif (name == 'LSTM_128_64_Dense_64_1'):
		hParams['LSTMLayers'] = [
		{
			'LSTM_numLayers': 128, 
			'return_sequences': 1,
			'LSTM_act': 'relu', 
		},
		{
			'LSTM_numLayers': 64, 
			'return_sequences': 0,
			'LSTM_act': 'relu', 
		},
	]
		hParams['denseLayers'] = [64, 1]
		hParams['optimizer'] = 'adam'

	elif (name == 'LSTM_128_128_Dense_64_1'):
		hParams['LSTMLayers'] = [
		{
			'LSTM_numLayers': 128, 
			'return_sequences': 1,
			'LSTM_act': 'relu', 
		},
		{
			'LSTM_numLayers': 128, 
			'return_sequences': 0,
			'LSTM_act': 'relu', 
		},
	]
		hParams['denseLayers'] = [64, 1]
		hParams['optimizer'] = 'adam'	

	elif (name == 'LSTM_256_128_Dense_64_1'):
		hParams['LSTMLayers'] = [
		{
			'LSTM_numLayers': 256, 
			'return_sequences': 1,
			'LSTM_act': 'relu', 
		},
		{
			'LSTM_numLayers': 128, 
			'return_sequences': 0,
			'LSTM_act': 'relu', 
		},
	]
		hParams['denseLayers'] = [64, 1]
		hParams['optimizer'] = 'adam'	

	elif (name == 'LSTM_256_128_Dense_64_64_1'):
		hParams['LSTMLayers'] = [
		{
			'LSTM_numLayers': 256, 
			'return_sequences': 1,
			'LSTM_act': 'relu', 
		},
		{
			'LSTM_numLayers': 128, 
			'return_sequences': 0,
			'LSTM_act': 'relu', 
		},
	]
		hParams['denseLayers'] = [64, 64, 1]
		hParams['optimizer'] = 'adam'	

	elif (name == 'LSTM_256_128_Dense_64_32_1'):
		hParams['LSTMLayers'] = [
		{
			'LSTM_numLayers': 256, 
			'return_sequences': 1,
			'LSTM_act': 'relu', 
		},
		{
			'LSTM_numLayers': 128, 
			'return_sequences': 0,
			'LSTM_act': 'relu', 
		},
	]
		hParams['denseLayers'] = [64, 32, 1]
		hParams['optimizer'] = 'adam'	

	elif (name == 'LSTM_64_Dense_128_128_64_1'):
		hParams['LSTMLayers'] = [
		{
			'LSTM_numLayers': 64, 
			'return_sequences': 0,
			'LSTM_act': 'relu', 
		},
	]
		hParams['denseLayers'] = [128, 128, 64, 1]
		hParams['optimizer'] = 'adam'	

	elif (name == 'LSTM_64_Dense_256_128_64_1'):
		hParams['LSTMLayers'] = [
		{
			'LSTM_numLayers': 64, 
			'return_sequences': 0,
			'LSTM_act': 'relu', 
		},
	]
		hParams['denseLayers'] = [256, 128, 64, 1]
		hParams['optimizer'] = 'adam'	

	elif (name == 'LSTM_64_64_Dense_128_128_64_1'):
		hParams['LSTMLayers'] = [
		{
			'LSTM_numLayers': 64, 
			'return_sequences': 1,
			'LSTM_act': 'relu', 
		},
		{
			'LSTM_numLayers': 64, 
			'return_sequences': 0,
			'LSTM_act': 'relu', 
		},
	]
		hParams['denseLayers'] = [128, 128, 64, 1]
		hParams['optimizer'] = 'adam'	

	elif (name == 'LSTM_64_32_Dense_128_128_64_1'):
		hParams['LSTMLayers'] = [
		{
			'LSTM_numLayers': 64, 
			'return_sequences': 1,
			'LSTM_act': 'relu', 
		},
		{
			'LSTM_numLayers': 32, 
			'return_sequences': 0,
			'LSTM_act': 'relu', 
		},
	]
		hParams['denseLayers'] = [128, 128, 64, 1]
		hParams['optimizer'] = 'adam'	

	elif (name == 'LSTM_128_64_Dense_256_128_64_1'):
		hParams['LSTMLayers'] = [
		{
			'LSTM_numLayers': 128, 
			'return_sequences': 1,
			'LSTM_act': 'relu', 
		},
		{
			'LSTM_numLayers': 64, 
			'return_sequences': 0,
			'LSTM_act': 'relu', 
		},
	]
		hParams['denseLayers'] = [256, 128, 64, 1]
		hParams['optimizer'] = 'adam'	

	elif (name == 'LSTM_128_Dense_128_128_64_1'):
		hParams['LSTMLayers'] = [
		{
			'LSTM_numLayers': 128, 
			'return_sequences': 0,
			'LSTM_act': 'relu', 
		},
	]
		hParams['denseLayers'] = [256, 128, 64, 1]
		hParams['optimizer'] = 'adam'

	elif (name == 'LSTM_128_64_64_Dense_256_128_64_1'):
		hParams['LSTMLayers'] = [
		{
			'LSTM_numLayers': 128, 
			'return_sequences': 1,
			'LSTM_act': 'relu', 
		},
		{
			'LSTM_numLayers': 64, 
			'return_sequences': 1,
			'LSTM_act': 'relu', 
		},
		{
			'LSTM_numLayers': 64, 
			'return_sequences': 0,
			'LSTM_act': 'relu', 
		},
	]
		hParams['denseLayers'] = [256, 128, 64, 1]
		hParams['optimizer'] = 'adam'

	elif (name == 'LSTM_128_Dense_512_256_128_64_1'):
		hParams['LSTMLayers'] = [
		{
			'LSTM_numLayers': 128, 
			'return_sequences': 0,
			'LSTM_act': 'relu', 
		},
	]
		hParams['denseLayers'] = [512, 256, 128, 64, 1]
		hParams['optimizer'] = 'adam'

	elif (name == 'LSTM_128_64_Dense_512_256_128_64_1'):
		hParams['LSTMLayers'] = [
		{
			'LSTM_numLayers': 128, 
			'return_sequences': 1,
			'LSTM_act': 'relu', 
		},
		{
			'LSTM_numLayers': 64, 
			'return_sequences': 0,
			'LSTM_act': 'relu', 
		},
	]
		hParams['denseLayers'] = [512, 256, 128, 64, 1]
		hParams['optimizer'] = 'adam'

	elif (name == 'GRU_128_64_Dense_32_1'):
		hParams['GRULayers'] = [
		{
			'GRU_numLayers': 128, 
			'return_sequences': 1,
			'GRU_act': 'relu', 
		},
		{
			'GRU_numLayers': 64, 
			'return_sequences': 0,
			'GRU_act': 'relu', 
		},
	]
		hParams['denseLayers'] = [32, 1]
		hParams['optimizer'] = 'adam'

	elif (name == 'GRU_128_64_Dense_64_1'):
		hParams['GRULayers'] = [
		{
			'GRU_numLayers': 128, 
			'return_sequences': 1,
			'GRU_act': 'relu', 
		},
		{
			'GRU_numLayers': 64, 
			'return_sequences': 0,
			'GRU_act': 'relu', 
		},
	]
		hParams['denseLayers'] = [64, 1]
		hParams['optimizer'] = 'adam'

	elif (name == 'GRU_128_128_Dense_64_1'):
		hParams['GRULayers'] = [
		{
			'GRU_numLayers': 128, 
			'return_sequences': 1,
			'GRU_act': 'relu', 
		},
		{
			'GRU_numLayers': 128, 
			'return_sequences': 0,
			'GRU_act': 'relu', 
		},
	]
		hParams['denseLayers'] = [64, 1]
		hParams['optimizer'] = 'adam'	

	elif (name == 'GRU_256_128_Dense_64_1'):
		hParams['GRULayers'] = [
		{
			'GRU_numLayers': 256, 
			'return_sequences': 1,
			'GRU_act': 'relu', 
		},
		{
			'GRU_numLayers': 128, 
			'return_sequences': 0,
			'GRU_act': 'relu', 
		},
	]
		hParams['denseLayers'] = [64, 1]
		hParams['optimizer'] = 'adam'	

	elif (name == 'GRU_256_128_Dense_64_64_1'):
		hParams['GRULayers'] = [
		{
			'GRU_numLayers': 256, 
			'return_sequences': 1,
			'GRU_act': 'relu', 
		},
		{
			'GRU_numLayers': 128, 
			'return_sequences': 0,
			'GRU_act': 'relu', 
		},
	]
		hParams['denseLayers'] = [64, 64, 1]
		hParams['optimizer'] = 'adam'	

	elif (name == 'GRU_256_128_Dense_64_32_1'):
		hParams['GRULayers'] = [
		{
			'GRU_numLayers': 256, 
			'return_sequences': 1,
			'GRU_act': 'relu', 
		},
		{
			'GRU_numLayers': 128, 
			'return_sequences': 0,
			'GRU_act': 'relu', 
		},
	]
		hParams['denseLayers'] = [64, 32, 1]
		hParams['optimizer'] = 'adam'	

	elif (name == 'GRU_64_Dense_128_128_64_1'):
		hParams['GRULayers'] = [
		{
			'GRU_numLayers': 64, 
			'return_sequences': 0,
			'GRU_act': 'relu', 
		},
	]
		hParams['denseLayers'] = [128, 128, 64, 1]
		hParams['optimizer'] = 'adam'	

	elif (name == 'GRU_64_Dense_256_128_64_1'):
		hParams['GRULayers'] = [
		{
			'GRU_numLayers': 64, 
			'return_sequences': 0,
			'GRU_act': 'relu', 
		},
	]
		hParams['denseLayers'] = [256, 128, 64, 1]
		hParams['optimizer'] = 'adam'	

	elif (name == 'GRU_64_64_Dense_128_128_64_1'):
		hParams['GRULayers'] = [
		{
			'GRU_numLayers': 64, 
			'return_sequences': 1,
			'GRU_act': 'relu', 
		},
		{
			'GRU_numLayers': 64, 
			'return_sequences': 0,
			'GRU_act': 'relu', 
		},
	]
		hParams['denseLayers'] = [128, 128, 64, 1]
		hParams['optimizer'] = 'adam'	

	elif (name == 'GRU_64_32_Dense_128_128_64_1'):
		hParams['GRULayers'] = [
		{
			'GRU_numLayers': 64, 
			'return_sequences': 1,
			'GRU_act': 'relu', 
		},
		{
			'GRU_numLayers': 32, 
			'return_sequences': 0,
			'GRU_act': 'relu', 
		},
	]
		hParams['denseLayers'] = [128, 128, 64, 1]
		hParams['optimizer'] = 'adam'	

	elif (name == 'GRU_128_64_Dense_256_128_64_1'):
		hParams['GRULayers'] = [
		{
			'GRU_numLayers': 128, 
			'return_sequences': 1,
			'GRU_act': 'relu', 
		},
		{
			'GRU_numLayers': 64, 
			'return_sequences': 0,
			'GRU_act': 'relu', 
		},
	]
		hParams['denseLayers'] = [256, 128, 64, 1]
		hParams['optimizer'] = 'adam'	

	elif (name == 'GRU_128_Dense_128_128_64_1'):
		hParams['GRULayers'] = [
		{
			'GRU_numLayers': 128, 
			'return_sequences': 0,
			'GRU_act': 'relu', 
		},
	]
		hParams['denseLayers'] = [256, 128, 64, 1]
		hParams['optimizer'] = 'adam'

	elif (name == 'GRU_128_64_64_Dense_256_128_64_1'):
		hParams['GRULayers'] = [
		{
			'GRU_numLayers': 128, 
			'return_sequences': 1,
			'GRU_act': 'relu', 
		},
		{
			'GRU_numLayers': 64, 
			'return_sequences': 1,
			'GRU_act': 'relu', 
		},
		{
			'GRU_numLayers': 64, 
			'return_sequences': 0,
			'GRU_act': 'relu', 
		},
	]
		hParams['denseLayers'] = [256, 128, 64, 1]
		hParams['optimizer'] = 'adam'

	elif (name == 'GRU_128_Dense_512_256_128_64_1'):
		hParams['GRULayers'] = [
		{
			'GRU_numLayers': 128, 
			'return_sequences': 0,
			'GRU_act': 'relu', 
		},
	]
		hParams['denseLayers'] = [512, 256, 128, 64, 1]
		hParams['optimizer'] = 'adam'

	elif (name == 'GRU_128_64_Dense_512_256_128_64_1'):
		hParams['GRULayers'] = [
		{
			'GRU_numLayers': 128, 
			'return_sequences': 1,
			'GRU_act': 'relu', 
		},
		{
			'GRU_numLayers': 64, 
			'return_sequences': 0,
			'GRU_act': 'relu', 
		},
	]
		hParams['denseLayers'] = [512, 256, 128, 64, 1]
		hParams['optimizer'] = 'adam'

	elif (name == 'BiLSTM_128_64_Dense_32_1'):
		hParams['BiLSTMLayers'] = [
		{
			'BiLSTM_numLayers': 128, 
			'return_sequences': 1,
			'BiLSTM_act': 'relu', 
		},
		{
			'BiLSTM_numLayers': 64, 
			'return_sequences': 0,
			'BiLSTM_act': 'relu', 
		},
	]
		hParams['denseLayers'] = [32, 1]
		hParams['optimizer'] = 'adam'

	elif (name == 'BiLSTM_128_64_Dense_64_1'):
		hParams['BiLSTMLayers'] = [
		{
			'BiLSTM_numLayers': 128, 
			'return_sequences': 1,
			'BiLSTM_act': 'relu', 
		},
		{
			'BiLSTM_numLayers': 64, 
			'return_sequences': 0,
			'BiLSTM_act': 'relu', 
		},
	]
		hParams['denseLayers'] = [64, 1]
		hParams['optimizer'] = 'adam'

	elif (name == 'BiLSTM_128_128_Dense_64_1'):
		hParams['BiLSTMLayers'] = [
		{
			'BiLSTM_numLayers': 128, 
			'return_sequences': 1,
			'BiLSTM_act': 'relu', 
		},
		{
			'BiLSTM_numLayers': 128, 
			'return_sequences': 0,
			'BiLSTM_act': 'relu', 
		},
	]
		hParams['denseLayers'] = [64, 1]
		hParams['optimizer'] = 'adam'	

	elif (name == 'BiLSTM_256_128_Dense_64_1'):
		hParams['BiLSTMLayers'] = [
		{
			'BiLSTM_numLayers': 256, 
			'return_sequences': 1,
			'BiLSTM_act': 'relu', 
		},
		{
			'BiLSTM_numLayers': 128, 
			'return_sequences': 0,
			'BiLSTM_act': 'relu', 
		},
	]
		hParams['denseLayers'] = [64, 1]
		hParams['optimizer'] = 'adam'	

	elif (name == 'BiLSTM_256_128_Dense_64_64_1'):
		hParams['BiLSTMLayers'] = [
		{
			'BiLSTM_numLayers': 256, 
			'return_sequences': 1,
			'BiLSTM_act': 'relu', 
		},
		{
			'BiLSTM_numLayers': 128, 
			'return_sequences': 0,
			'BiLSTM_act': 'relu', 
		},
	]
		hParams['denseLayers'] = [64, 64, 1]
		hParams['optimizer'] = 'adam'	

	elif (name == 'BiLSTM_256_128_Dense_64_32_1'):
		hParams['BiLSTMLayers'] = [
		{
			'BiLSTM_numLayers': 256, 
			'return_sequences': 1,
			'BiLSTM_act': 'relu', 
		},
		{
			'BiLSTM_numLayers': 128, 
			'return_sequences': 0,
			'BiLSTM_act': 'relu', 
		},
	]
		hParams['denseLayers'] = [64, 32, 1]
		hParams['optimizer'] = 'adam'	

	elif (name == 'BiLSTM_64_Dense_128_128_64_1'):
		hParams['BiLSTMLayers'] = [
		{
			'BiLSTM_numLayers': 64, 
			'return_sequences': 0,
			'BiLSTM_act': 'relu', 
		},
	]
		hParams['denseLayers'] = [128, 128, 64, 1]
		hParams['optimizer'] = 'adam'	

	elif (name == 'BiLSTM_64_Dense_256_128_64_1'):
		hParams['BiLSTMLayers'] = [
		{
			'BiLSTM_numLayers': 64, 
			'return_sequences': 0,
			'BiLSTM_act': 'relu', 
		},
	]
		hParams['denseLayers'] = [256, 128, 64, 1]
		hParams['optimizer'] = 'adam'	

	elif (name == 'BiLSTM_64_64_Dense_128_128_64_1'):
		hParams['BiLSTMLayers'] = [
		{
			'BiLSTM_numLayers': 64, 
			'return_sequences': 1,
			'BiLSTM_act': 'relu', 
		},
		{
			'BiLSTM_numLayers': 64, 
			'return_sequences': 0,
			'BiLSTM_act': 'relu', 
		},
	]
		hParams['denseLayers'] = [128, 128, 64, 1]
		hParams['optimizer'] = 'adam'	

	elif (name == 'BiLSTM_64_32_Dense_128_128_64_1'):
		hParams['BiLSTMLayers'] = [
		{
			'BiLSTM_numLayers': 64, 
			'return_sequences': 1,
			'BiLSTM_act': 'relu', 
		},
		{
			'BiLSTM_numLayers': 32, 
			'return_sequences': 0,
			'BiLSTM_act': 'relu', 
		},
	]
		hParams['denseLayers'] = [128, 128, 64, 1]
		hParams['optimizer'] = 'adam'	

	elif (name == 'BiLSTM_128_64_Dense_256_128_64_1'):
		hParams['BiLSTMLayers'] = [
		{
			'BiLSTM_numLayers': 128, 
			'return_sequences': 1,
			'BiLSTM_act': 'relu', 
		},
		{
			'BiLSTM_numLayers': 64, 
			'return_sequences': 0,
			'BiLSTM_act': 'relu', 
		},
	]
		hParams['denseLayers'] = [256, 128, 64, 1]
		hParams['optimizer'] = 'adam'	

	elif (name == 'BiLSTM_128_Dense_128_128_64_1'):
		hParams['BiLSTMLayers'] = [
		{
			'BiLSTM_numLayers': 128, 
			'return_sequences': 0,
			'BiLSTM_act': 'relu', 
		},
	]
		hParams['denseLayers'] = [256, 128, 64, 1]
		hParams['optimizer'] = 'adam'

	elif (name == 'BiLSTM_128_64_64_Dense_256_128_64_1'):
		hParams['BiLSTMLayers'] = [
		{
			'BiLSTM_numLayers': 128, 
			'return_sequences': 1,
			'BiLSTM_act': 'relu', 
		},
		{
			'BiLSTM_numLayers': 64, 
			'return_sequences': 1,
			'BiLSTM_act': 'relu', 
		},
		{
			'BiLSTM_numLayers': 64, 
			'return_sequences': 0,
			'BiLSTM_act': 'relu', 
		},
	]
		hParams['denseLayers'] = [256, 128, 64, 1]
		hParams['optimizer'] = 'adam'

	elif (name == 'BiLSTM_128_Dense_512_256_128_64_1'):
		hParams['BiLSTMLayers'] = [
		{
			'BiLSTM_numLayers': 128, 
			'return_sequences': 0,
			'BiLSTM_act': 'relu', 
		},
	]
		hParams['denseLayers'] = [512, 256, 128, 64, 1]
		hParams['optimizer'] = 'adam'

	elif (name == 'BiLSTM_128_64_Dense_512_256_128_64_1'):
		hParams['BiLSTMLayers'] = [
		{
			'BiLSTM_numLayers': 128, 
			'return_sequences': 1,
			'BiLSTM_act': 'relu', 
		},
		{
			'BiLSTM_numLayers': 64, 
			'return_sequences': 0,
			'BiLSTM_act': 'relu', 
		},
	]
		hParams['denseLayers'] = [512, 256, 128, 64, 1]
		hParams['optimizer'] = 'adam'
		
	return hParams