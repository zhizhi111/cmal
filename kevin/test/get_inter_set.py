import bert_base_cased_vocab

if __name__ == '__main__':
    # 分词词典： key 为单词, value 为序号
    bert_vocab = {}
    # 实体 tag 词典： key 为 tag ，value 为序号
    object_vocab = {}

    f = open("bert-base-cased-vocab.txt", 'r', encoding='UTF-8')
    line = f.readline()
    i = 0
    while line:
        bert_vocab[line.replace('\n', '')] = i
        i = i + 1
        line = f.readline()
    f.close()

    f = open("objects_vocab.txt", 'r', encoding='UTF-8')
    line = f.readline()
    j = 0
    while line:
        object_vocab[(line.replace('\n', ''))] = j
        j = j + 1
        line = f.readline()
    f.close()

    map = {}
    for tag in object_vocab.keys():
        value_list = set()
        tag_split = tag.split(" ")
        tag_split_s = [s + "s" for s in tag_split]
        tag_split_es = [s + "es" for s in tag_split]
        tag_split_no_s = [s[0:-1] if s.endswith("s") else s for s in tag_split]
        tag_split_no_es = [s[0:-2] if s.endswith("es") else s for s in tag_split]
        for i in range(len(tag_split)):
            if tag_split[i] in bert_vocab.keys():
                value_list.add(bert_vocab[tag_split[i]])
            if tag_split_s[i] in bert_vocab.keys():
                value_list.add(bert_vocab[tag_split_s[i]])
            if tag_split_es[i] in bert_vocab.keys():
                value_list.add(bert_vocab[tag_split_es[i]])
            if tag_split_no_s[i] in bert_vocab.keys():
                value_list.add(bert_vocab[tag_split_no_s[i]])
            if tag_split_no_es[i] in bert_vocab.keys():
                value_list.add(bert_vocab[tag_split_no_es[i]])
        if len(value_list) > 0:
            map[object_vocab[tag]] = list(value_list)

            # print(tag, ':', [bert_base_cased_vocab.vocab[str(k)] for k in value_list])

    print('obj2bert:\n', map)
    print('len(map) = ', len(map))

    # inter_keys = bert_vocab.keys() & object_vocab.keys()
    # sub_keys = object_vocab.keys() - bert_vocab.keys()
    #
    # print('len(bert_vocab)=', len(bert_vocab), '\nlen(object_vocab)=', len(object_vocab), '\nlen(inter_keys)=',
    #       len(inter_keys))
    #
    # map = dict(zip([object_vocab[k] for k in inter_keys], [bert_vocab[k] for k in inter_keys]))
    # print('obj2bert:\n', map)
