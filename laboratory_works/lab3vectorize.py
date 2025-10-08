import numpy as np

def vectorize_sequence(data_vector, number_range = 300):
    result_matrix = np.zeros((len(data_vector), number_range - 1))
    for i, value in enumerate(data_vector):
        result_matrix[i, value] = 1
    return result_matrix

input_file = 'input_vector.npy'

input_vector = np.random.rand(1000, 1) * 300 + 1
input_vector = input_vector.astype('int8')
np.save(input_file, input_vector)

input_vector = np.load(input_file, allow_pickle=True)

output_matrix = vectorize_sequence(input_vector)
print(output_matrix)
np.save('output_matrix.npy', output_matrix)


