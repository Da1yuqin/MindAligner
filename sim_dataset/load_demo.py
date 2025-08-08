import json

file_path = './v2subj1257/category_image_idx_subj1.json'

with open(file_path, 'r') as file:
    data = json.load(file)

person_int_list = []

for item in data['person']:
    image_idx, sub_idx = map(int, item.split('_'))
    person_int_list.append([image_idx, sub_idx])

print(person_int_list)