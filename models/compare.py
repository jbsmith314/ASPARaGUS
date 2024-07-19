import sys
import torch
import time

def reduce_dimension(weights1, weights2):
        reduced1 = weights1[0]
        reduced2 = weights2[0]
        for index in range(1, len(weights1)):
                reduced1 = torch.cat((reduced1, weights1[index]))
                reduced2 = torch.cat((reduced2, weights2[index]))
        return reduced1, reduced2

def reduce_dimension_list(weights1, weights2):
        reduced1 = weights1[0]
        reduced2 = weights2[0]
        for index in range(1, len(weights1)):
                reduced1 += weights1[index]
                reduced2 += weights2[index]
        return reduced1, reduced2


if __name__ == "__main__":
        model1 = torch.load(sys.argv[1])
        model2 = torch.load(sys.argv[2])

        start_time = time.time()
        for index, key in enumerate(model1.keys()):
                weights1 = model1[key]
                weights2 = model2[key]
                for _ in range(max(0, len(model1[key].size()) - 1)):
                        weights1, weights2 = reduce_dimension(weights1, weights2)
                if index == 0:
                        all_weights1 = weights1
                        all_weights2 = weights2
                elif len(model1[key].size()) - 1 >= 0:
                        all_weights1 = torch.cat((all_weights1, weights1))
                        all_weights2 = torch.cat((all_weights2, weights2))

        abs_diff_1 = 0
        for index in range(len(all_weights1)):
                abs_diff_1 += abs(all_weights2[index] - all_weights1[index])

        end_time = time.time()
        print(f'The absolute difference is {abs_diff_1} between the two models with a size of {len(all_weights1)} parameters')
        print(f'This is an average change of {abs_diff_1 / len(all_weights1)} for each weight')
        print(f'Took {end_time - start_time} seconds')

        start_time = time.time()

        all_weights3 = []
        all_weights4 = []
        for index, key in enumerate(model1.keys()):
                weights3 = model1[key].tolist()
                weights4 = model2[key].tolist()
                for _ in range(max(0, len(model1[key].size()) - 1)):
                        weights3, weights4 = reduce_dimension_list(weights3, weights4)
                if len(model1[key].size()) - 1 >= 0:
                        all_weights3 += weights3
                        all_weights4 += weights4

        abs_diff_2 = 0
        for index in range(len(all_weights3)):
                abs_diff_2 += abs(all_weights4[index] - all_weights3[index])

        end_time = time.time()
        print(f'The absolute difference is {abs_diff_2} between the two models with a size of {len(all_weights3)} parameters')
        print(f'This is an average change of {abs_diff_2 / len(all_weights3)} for each weight')
        print(f'Took {end_time - start_time} seconds')

        for index in range(len(all_weights1)):
                if all_weights1[index] != all_weights3[index]:
                        print(f'Found difference between 1 and 3 at index {index}')
                if all_weights2[index] != all_weights4[index]:
                        print(f'Found difference between 2 and 4 at index {index}')

        for index in range(len(all_weights1)):
                print('Index', f'{index}:', all_weights1[index] - all_weights3[index], all_weights2[index] - all_weights4[index])

