import re
from collections import Counter
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# 红楼梦的章回数
CHAPTER_NUM = 120

# 将红楼梦全书按回目切分，并保存在data目录中
def cut_book_to_chapter(book_path, save_path_prefix):
  chapter_begin_pattern = '[('
  chapter_end_pattern = '(本章完)'
  
  #import ipdb; ipdb.set_trace()
  with open(book_path, 'r', encoding = 'utf-8') as f:
    book = f.read()

  beg_pos =  book.find(chapter_begin_pattern)
  chapter_index = 1
  while beg_pos != -1:
    end_pos = book.find(chapter_end_pattern, beg_pos)
    end_pos += len(chapter_end_pattern)

    current_chapter = book[beg_pos:end_pos+1]
    current_chapter_path = save_path_prefix + str(chapter_index)
    with open(current_chapter_path, 'w', encoding = 'utf-8') as f:
      f.write(current_chapter)
    
    beg_pos = book.find(chapter_begin_pattern, end_pos)
    chapter_index += 1

# 对每一章节预处理
def preprocess_chapter(raw_text):
  # 去除第一行和最后一行
  pos1 = raw_text.find('\n')
  pos2 = raw_text.rfind('\n')
  chapter = raw_text[pos1:pos2].strip()
  # 去除空格和换行符号
  chapter = re.sub('[\s]', '', chapter)
  
  return chapter

# 找到每一回都出现的字，即停用字
def get_stop_chars(dir_prefix):
  stop_chars = set()
  for i in range(1, CHAPTER_NUM+1):
    file_path = dir_prefix + str(i)
    # 读取章节文本
    with open(file_path, 'r', encoding = 'utf-8') as f:
      raw_text = f.read().strip()
    # 预处理
    chapter = preprocess_chapter(raw_text)
    # 求每个章节的交集 
    if i == 1:
      stop_chars = set(chapter)
    else:
      stop_chars &= set(chapter)

  return list(stop_chars)

# 求出每个停用字的字频，保存到向量中
def convert_chapter_to_vector(chapter, stop_chars):
  # 得到每个字的出现次数
  char_counter = Counter(chapter)
  # 得到该回目总字数 
  chapter_char_num = sum(char_counter.values())
  # 当前回目的特征向量
  feature_vector = np.zeros(len(stop_chars), dtype='float32')
  for i, c in enumerate(stop_chars):
    feature_vector[i] = char_counter[c]
  feature_vector /= chapter_char_num
  
  return feature_vector

# 将每一章节的向量作为一行，构成矩阵
def convert_book_to_matrix(dir_prefix, stop_chars):
  observations = np.zeros((CHAPTER_NUM, len(stop_chars)), dtype='float32')

  for i in range(1, CHAPTER_NUM+1):
    file_path = dir_prefix + str(i)
    # 读取章节文本
    with open(file_path, 'r', encoding = 'utf-8') as f:
      raw_text = f.read().strip()
    # 预处理
    chapter = preprocess_chapter(raw_text)
    # 得到当前回目的向量
    observations[i-1, :] = convert_chapter_to_vector(chapter, stop_chars) 

  return observations

# 降维到3维，并可视化
def scatters_in_3d(samples, is_labelled = False):
  # PCA 降维到2维便于可视化
  pca = PCA(n_components=3)
  reduced_data = pca.fit_transform(samples)

  fig = plt.figure()
  ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=9, azim=-170)
  for c,  rng in [('r', (0, 80)), ('b', (80, 120))]:
    xs = reduced_data[rng[0]:rng[1], 0]
    ys = reduced_data[rng[0]:rng[1], 1]
    zs = reduced_data[rng[0]:rng[1], 2]
    ax.scatter(xs, ys, zs, c=c)


  ax.w_xaxis.set_ticklabels([])
  ax.w_yaxis.set_ticklabels([])
  ax.w_zaxis.set_ticklabels([])

  if is_labelled:
    for ix in np.arange(len(samples)):
      ax.text(reduced_data[ix, 0], reduced_data[ix, 1],reduced_data[ix, 2],
          str(ix+1), verticalalignment='center', fontsize=10)

  plt.show()

# 构建训练数据并用kNN分类器分类
def knn_clf(observations, n_neighbors):
  # 构建训练数据
  range1 = [20, 30]
  len1 = len(range(range1[0], range1[1]))
  range2 = [110, 120]
  len2 = len(range(range2[0], range2[1]))

  training_index = list(range(range1[0], range1[1])) + list(range(range2[0],
    range2[1]))
  training_data = observations[training_index, :]
  training_label = np.ones(len1+len2, dtype='int32')
  training_label[len1:] = 2
  # 最近邻分类器
  knn = KNeighborsClassifier(n_neighbors = 3)#, weights = 'distance')
  knn.fit(training_data, training_label) 
  # 预测
  knn_pre = knn.predict(observations)

  print('第一回至第八十回')
  for i in range(8):
    print(knn_pre[i*10:(i+1)*10])

  print('第八十一回至第一百二十回')
  for i in range(8,12):
    print(knn_pre[i*10:(i+1)*10])

if __name__ == '__main__':
  chapter_prefix = 'data/chapter-'
  # 将红楼梦全书分回目存储
  cut_book_to_chapter('./data/dream_of_red_chamber.txt', chapter_prefix)

  # 获取每个章节都出现过的字
  stop_chars = get_stop_chars(chapter_prefix)
  # 将全书转换为特征矩阵
  observations = convert_book_to_matrix(chapter_prefix, stop_chars)
  # 降维并画图
  #scatters_in_3d(observations, True)
  # kNN分类
  knn_clf(observations, 3)
  #import ipdb; ipdb.set_trace()
