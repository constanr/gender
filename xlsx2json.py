import xlrd
from collections import OrderedDict
import simplejson as json

wb = xlrd.open_workbook('data/blog-gender-dataset.xlsx')
sh = wb.sheet_by_index(0)

posts_list = []

for row in range(1, sh.nrows):
    post = OrderedDict()
    row_values = sh.row_values(row)
    post['text'] = row_values[0]
    post['gender'] = row_values[1]

    posts_list.append(post)

j = json.dumps(posts_list)

with open('data/blog-gender-dataset.json', 'w') as f:
    f.write(j)
