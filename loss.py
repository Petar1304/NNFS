import numpy as np

softmax_outputs = np.array([[0.7, 0.1, 0.2],
                            [0.1, 0.5, 0.4],
                            [0.02, 0.9, 0.08]])


class_targets = [0, 1, 1]

# for index, distribution in zip(class_targets, softmax_outputs):
# print(softmax_outputs[[0, 1, 2], class_targets])
# ([first_dimension_indexes], [second_dimension_indexes])
# will give us [0.7, 0.5, 0.9]

neg_log = -np.log(softmax_outputs[[0, 1, 2], class_targets])

neg_log2 = -np.log(softmax_outputs[range(len(softmax_outputs)), class_targets])


average_loss = np.mean(neg_log)
print(average_loss)


average_loss2 = np.mean(neg_log2)
print(average_loss2)







