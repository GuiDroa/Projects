"""This is a sample file for hw1. 
It contains the function that should be submitted,
except all it does is output a random value out of the
possible values that are allowed.
- Dr. Licato"""

import re
import random

def problem1(NPs, s):
    # Let all noun phrases in NPS lowercase
    NPs = set(np.lower() for np in NPs) #Sets the words in the initial "set"(NPs) to lower case.

    # Matches any word in NPs
    NPs_words = r'\b(?:' + '|'.join(re.escape(np) for np in NPs) + r')\b'

    # Regex parterns is divided into two parts. Part one includes all was, were, is, are, a type of, etc (b, a) relationships
    # Part two includes the (a, b) (a, c) ... relationships. The words including, such as. It also includes the comma. 
    # The {NPs_words} ensures that the word being caught is from the initial set of words.
    pattern = (
        rf'({NPs_words})\s+(?:was|were|is|are)\s*(?:a|an)?\s+(?:a\s+type\s+of|a\s+kind\s+of)?\s*({NPs_words})'
        r'|'
        rf'({NPs_words})\s*(?:,?\s+(?:including|such\s+as)\s+)((?:{NPs_words}(?:,\s*)?)+)(?:\s*(?:and|or)\s+({NPs_words}))?'
    )
    
    # The regex is divided into 5 groups, the rest is ignored using ?: . 

    # Find the matches
    matches = re.findall(pattern, s, re.IGNORECASE)
    
    # Creates empty set for storing hypernyms
    hypernyms = set()
    for m in matches:

        # m[0] -> first captured group
        # m[1] -> second captured group
        # m[2] -> third captured group
        # m[3] -> fourth captured group
        # m[4] -> fifth captured group (Only captures it when there is an and/ or)

        # If a match happen within group 0, it means that the first regex is used
        if m[0]: #m[1] is the second group that captures the NPs word.
            b = m[1].lower() # Ensures both a and b are lowercase
            a = m[0].lower()
            if a in NPs and b in NPs: # Ensures both a and b are present in NPs words
                hypernyms.add((b, a)) # 

        # If a match happen within group 2, it means that the second regex is used
        elif m[2]:
            a = m[2].lower()
            for b in m[3].split(','): # Split m[3] commas
                b = b.strip().lower()
                if a in NPs and b in NPs: # if a and b are present in NPs (checks with all words being lowercase)
                    hypernyms.add((a, b)) # add a, b to the set, following the order (a , b)
            if m[4]: # if m[4] is captured, it means that there is an and/ or. Therefore, second regex.
                d = m[4].lower() #Ensures is lower case. a (m[2]) is already ensured to be lower case.
                if a in NPs and d in NPs: # Ensures both a and d are present in NPS words.
                    hypernyms.add((a, d))  # Adds a and d to the hypernyms.

    return hypernyms

# Examples used for testing

# s =  "Hemingway was an author of many classics. But also, Hemingway was a bibliophile, having read the works of every other famous American author, such as William Faulkner and Mark Twain."
# NPs = ['hemingway', 'bibliophile', 'author', 'william faulkner', 'mark twain']
# print(problem1(NPs, s))  

# s =  """All mammals, such as dogs, birds, vasco and cats, eat to survive. Mammals are living things, aren't they? Where the big gorilla is king kong sarrada"""
# NPs = ['dogs', 'cats', 'vasco','mammals', 'living things', 'birds','big gorilla','king kong']
# print(problem1(NPs, s))  

# s =  "Some animals, including cats, are considered. But it is NOT true that dogs are animals ; I refuse to accept it. Dogs, though? Those are types of cats. Also, puppies are dogs."
# NPs = ['animals', 'dogs', 'cats']
# print(problem1(NPs, s))  


def problem2(s1, s2):

    str1 =  len(s1)
    str2 = len(s2)

    if not str2: # Checks if s2 is an empty string
        return str1 # Return the length of s1

    if str1 < str2: #Ensures that string s1 is always longer than s2
        s1, s2 = s2, s1 # Switches s1 with s2 and s2 with s1
    
    dist_prev = list(range(str2 + 1)) # initialize dist_prev

    for idx1, char1 in enumerate(s1):  # For loop interates over every char on string saving index number as idx1.
        dist_cur = [idx1 + 1] # initialize variable as i + 1
        for idx2, char2 in enumerate(s2): # For loop interates over every char on string saving index number as idx2. Used to compare s1 and s2 characters
            cost_insert = dist_prev[idx2 + 1] + 1 # Formula for inserting
            cost_delete = dist_cur[idx2] + 1      # Formula for deleting
            if (char1 == char2):                  # Formula for substitution -> There is 2 options.
                cost_subs = dist_prev[idx2]  # If equal characters, there is no need to add, therefore copy from dist_prev
            else:
                cost_subs = dist_prev[idx2] + 2 # If characters are not equal, we add +2 to the value of dist_prev

            dist_cur.append(min(cost_insert, cost_delete, cost_subs)) # Adds the lowest of the three formula compute before to the dist_cur
        dist_prev = dist_cur # Updates dist_prev to be dist_cur for the next loop interation
    
    edit_distance = dist_prev[-1]
    return edit_distance # return the "Top rightmost value". Min cost of edit distance

