import random

def generate_random_data(file_name, cut, n_random):

    file_in = open(file_name, 'r')

    file_in_lines = file_in.readlines()

    property_value_list = []
    descriptor_list = []

    for i in range(len(file_in_lines)):
        property_value = float(file_in_lines[i].split()[-1])
        descriptor = file_in_lines[i].split()[:-1]

        tmp03 ="    ".join(str(float(i)) for i in descriptor)
        #if property_value <= cut:
        if property_value >= cut:
           descriptor_list.append(descriptor)
           property_value_list.append(property_value)


    # random
    random_indexs = random.sample(range(0, len(descriptor_list)), n_random)

    print("random_indexes", random_indexs)

    file_name = "regression_data_" + str(n_random) + ".txt"
    out_file = open(file_name, 'w')

    for iIndex in random_indexs:
        #out_file.write((descriptor_list[iIndex], property_value_list[iIndex]))
        tmp04 ="    ".join(str(float(i)) for i in descriptor_list[iIndex])
        #print(descriptor_list[iIndex], property_value_list[iIndex], file=out_file)
        print(tmp04, "         ", property_value_list[iIndex], file=out_file)

    out_file.close()

generate_random_data("all_data_MOF.txt", cut=-10.0, n_random=60)
