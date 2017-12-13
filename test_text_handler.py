from text_handler import split_input_into_chars, split_input_into_words, split_input_into_sentences

test_inputs = ["========This..is a  test\tof. the,so called>>>> \"input\" Function!!!1!", "This\t\t\tis<another,,sentence"]

char_output = split_input_into_chars(test_inputs)
print(char_output)
word_output = split_input_into_words(test_inputs)
print(word_output)
