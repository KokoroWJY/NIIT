from nltk.corpus import gutenberg

for fileld in gutenberg.fileids():
    raw = gutenberg.raw(fileld)
    num_chars = len(raw)
    words = gutenberg.words(fileld)
    num_words = len(words)
    sents = gutenberg.sents(fileld)
    num_sents = len(sents)
    print('%d %d %d %s' % (num_chars, num_words, num_sents, fileld))
