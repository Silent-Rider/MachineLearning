import numpy as np

def vector_to_binary_matrix(data_vector):
    data = np.array(data_vector)
    if data.min() < 0:
        raise ValueError("Data must be positive")

    max_val = data.max()
    if max_val == 0:
        num_bits = 1
    else:
        num_bits = int(np.floor(np.log2(max_val)) + 1)

    powers = 2 ** np.arange(num_bits - 1, -1, -1)
    binary_matrix = (data[:, None] & powers) > 0
    return binary_matrix.astype(np.uint8)

input_file = 'input_vector.npy'

power = 8
#Создание входного вектора
input_vector = np.random.rand(1000) * (2 ** power)
input_vector = input_vector.astype('uint8')
np.save(input_file, input_vector)

input_vector = np.load(input_file)
print(input_vector[:5])

output_matrix = vector_to_binary_matrix(input_vector)
print(output_matrix[:3])
np.save('output_matrix.npy', output_matrix)


