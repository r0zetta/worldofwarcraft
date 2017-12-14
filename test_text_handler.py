from text_handler import split_input_into_chars, split_input_into_words, split_input_into_sentences

test_inputs = ["========This..is a  test\tof. the,so called>>>> \"input\" Function!!!1!", "This\t\t\tis<another,,sentence", "0ne<< moAr> ! sen<tence"]

print("Char splitting test")
print("===================")
char_output = split_input_into_chars(test_inputs)
print(char_output)
print("Original sentences")
for t in test_inputs:
    print t
print("Reassembled")
print ''.join(char_output)
print
word_output = split_input_into_words(test_inputs)
print(word_output)
print("Original sentences")
for t in test_inputs:
    print t
print("Reassembled")
print ''.join(word_output)
print
