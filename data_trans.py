from train import parse_args
import json

args = parse_args()

# 定义输入文件和输出文件的文件名
# input_filename = 'ent_ids_2'
# output_filename = 'ent_ids_2_seq'
# output_filename_dict_name = 'num_dict'


# input = args.data + '/' + args.lang + '/' + input_filename
# output = args.data + '/' + args.lang + '/' + output_filename
# output_dict = args.data + '/' + args.lang + '/' + output_filename_dict_name
# # 打开输入文件进行读取
# # with open(input, 'r', encoding='utf-8') as infile:
# #     # 打开输出文件准备写入
# #     with open(output, 'w', encoding='utf-8') as outfile:
# #         # 逐行读取输入文件
# #         for line in infile:
# #             # 将每行中的逗号替换为制表符
# #             new_line = line.replace(',', '\t')
# #             # 将替换后的行写入输出文件
# #             outfile.write(new_line)
# # 打开输入文件进行读取
# line_number_dict = {}
# with open(input, 'r', encoding='utf-8') as infile, \
#         open(output, 'w', encoding='utf-8') as outfile, \
#         open(output_dict, 'a', encoding='utf-8') as dictfile:
#     # 逐行读取输入文件
#     for index, line in enumerate(infile, start=19388):
#         parts = line.strip().split('\t')
#
#         # 检查行是否为空或是否有足够的分割部分
#         if parts:
#             # 将行号和行的第一个内容写入输出文件
#             outfile.write(f"{index}\t{parts[1]}\n")
#             dictfile.write(f"{index}\t{parts[0]}\n")
#
#             # 将行号和行的第一个内容添加到字典中
#             # line_number_dict[index] = parts[0]
#
#         # 将字典写入另一个输出文件
#     # for key, value in line_number_dict.items():
#     #     dictfile.write(f"{key}: {value}\n")
# input_dict_name = 'num_dict'
# input_filename = 'ref_ent_ids'
# output_filename = 'ref_ent_ids_seq'
#
# input = args.data + '/' + args.lang + '/' + input_filename
# output = args.data + '/' + args.lang + '/' + output_filename
# input_dict = args.data + '/' + args.lang + '/' + input_dict_name
# num_dict = {}
# with open(input, 'r', encoding='utf-8') as infile, \
#     open(input_dict, 'r', encoding='utf-8') as inputDict, \
#             open(output, 'w', encoding='utf-8') as outfile:
#     for line in inputDict:
#         part = line.strip().split('\t')
#         num_dict[part[1]] = part[0]
#     for line in infile:
#         part = line.strip().split('\t')
#         outfile.write(f"{num_dict[part[0]]}\t{num_dict[part[1]]}\n")

# input_dict_name = 'num_dict'
# input_filename = 'triples_2'
# output_filename = 'triples_2_seq'
#
# input = args.data + '/' + args.lang + '/' + input_filename
# output = args.data + '/' + args.lang + '/' + output_filename
# input_dict = args.data + '/' + args.lang + '/' + input_dict_name
# num_dict = {}
# with open(input, 'r', encoding='utf-8') as infile, \
#     open(input_dict, 'r', encoding='utf-8') as inputDict, \
#             open(output, 'w', encoding='utf-8') as outfile:
#     for line in inputDict:
#         part = line.strip().split('\t')
#         num_dict[part[1]] = part[0]
#     for line in infile:
#         part = line.strip().split('\t')
#         outfile.write(f"{num_dict[part[0]]}\t{part[1]}\t{num_dict[part[2]]}\n")
input_dict_name = 'num_dict'
input_filename = 'zh_vectorList.json'
output_filename = 'zh_vectorList_seq.json'

input = args.data + '/' + args.lang + '/' + input_filename
output = args.data + '/' + args.lang + '/' + output_filename
input_dict = args.data + '/' + args.lang + '/' + input_dict_name
num_dict = {}
# with open(input, 'r', encoding='utf-8') as infile, \
#     open(input_dict, 'r', encoding='utf-8') as inputDict, \
#             open(output, 'w', encoding='utf-8') as outfile:
#     for line in inputDict:
#         part = line.strip().split('\t')
#         num_dict[part[1]] = part[0]
#     data = json.load(infile)
#     print(len(data), len(data[0]), len(num_dict))
#     output_data = []
#     for key in num_dict:
#         output_data.append(data[int(key)])
#     json.dump(output_data, outfile, ensure_ascii=False, indent=4)




