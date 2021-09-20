# 3119009475-
import re
import html
import jieba
import jieba.analyse
from sklearn.metrics.pairwise import cosine_similarity


class CosineSimilarity(object):
    # 余弦相似度
    def __init__(self, x, y):
        self.s1 = x
        self.s2 = y

    @staticmethod
    def extract_keyword(content):  # 提取关键词
        # 正则过滤 html 标签
        re_exp = re.compile(r'(<style>.*?</style>)|(<[^>]+>)', re.S)
        content = re_exp.sub(' ', content)
        # html 转义符实体化
        content = html.unescape(content)
        # 切割
        seg = [i for i in jieba.cut(content, cut_all=True) if i != '']
        # 提取关键词
        keywords = jieba.analyse.extract_tags("|".join(seg), topK=200, withWeight=False)
        return keywords

    @staticmethod
    def one_hot(word_dict, keywords):  # oneHot编码
        # cut_code = [word_dict[word] for word in keywords]
        cut_code = [0] * len(word_dict)
        for word in keywords:
            cut_code[word_dict[word]] += 1
        return cut_code

        # 去除停用词
        jieba.analyse.set_stop_words(r'E:\PyCharm\test\stopwords.txt')

    def main(self):

        # 提取关键词
        keywords1 = self.extract_keyword(self.s1)
        keywords2 = self.extract_keyword(self.s2)
        # 词的并集
        union = set(keywords1).union(set(keywords2))
        # 编码
        word_dict = {}
        i = 0
        for word in union:
            word_dict[word] = i
            i += 1
        # oneHot编码
        s1_cut_code = self.one_hot(word_dict, keywords1)
        s2_cut_code = self.one_hot(word_dict, keywords2)
        # 余弦相似度计算
        sample = [s1_cut_code, s2_cut_code]
        # 除零处理
        try:
            sim = cosine_similarity(sample)
            return sim[1][0]
        except Exception as e:
            print(e)
            return 0.0

# 测试
if __name__ == '__main__':
    with open(r'E:\PyCharm\test\orig.txt', 'r', encoding='utf-8') as x, open(r'E:\PyCharm\test\orig_0.8_dis_15.txt', 'r',
                                                                       encoding='utf-8') as y:
        content_x = x.read()
        content_y = y.read()
        similarity = CosineSimilarity(content_x, content_y)
        similarity = similarity.main()
        print('相似度: %.2f%%' % (similarity * 100))
        f3 = open(r'E:\PyCharm\test\result.txt', 'w')
        f3.write('文本相似度为:' + str(similarity * 100) + '%')  # 将结果转换为字符串类型并写入result.txt文件
