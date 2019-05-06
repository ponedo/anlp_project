import os
import re
import pickle
from collections import Counter
import spacy
from nltk import word_tokenize, sent_tokenize
from utils import *


regexes_needed=(
            r"(?:[0-9]+\s[A-Za-z0-9\.]*\.[A-Za-z0-9\.]*\s[0-9]+(?:\s,\s[0-9\-]+)?(?:\s\([^\(\)]*\))?)",  # case number
            r"(?:col\.\s[0-9]+,\sl+\.\s[0-9-]+)", # col. 1, ll. 10-31
            r"(?:Nos?\.(?:\s[0-9-]+)+)", # No. 123-3
            r"(?:[\′\'][0-9]+\s[Pp]atent)", # ′411 patent
            r"(?:Id.\sat\s[])", # Id. at 13
            r"(?:[(]r[)])", # (r)
            r"(?:Jan(?:\.|uary)\s[0-9]{1,2},\s[0-9]{4})", #date
            r"(?:Feb(?:\.|ruary)\s[0-9]{1,2},\s[0-9]{4})", #date
            r"(?:Mar\.\s[0-9]{1,2},\s[0-9]{4})", #date
            r"(?:March\s[0-9]{1,2},\s[0-9]{4})", #date
            r"(?:Apr\.\s[0-9]{1,2},\s[0-9]{4})", #date
            r"(?:April\s[0-9]{1,2},\s[0-9]{4})", #date
            r"(?:May\s[0-9]{1,2},\s[0-9]{4})", #date
            r"(?:Jun\.\s[0-9]{1,2},\s[0-9]{4})", #date
            r"(?:June\s[0-9]{1,2},\s[0-9]{4})", #date
            r"(?:Jul\.\s[0-9]{1,2},\s[0-9]{4})", #date
            r"(?:July\s[0-9]{1,2},\s[0-9]{4})", #date
            r"(?:Aug(?:\.|ust)\s[0-9]{1,2},\s[0-9]{4})", #date
            r"(?:Sept(?:\.|ember)\s[0-9]{1,2},\s[0-9]{4})", #date
            r"(?:Oct(?:\.|ober)\s[0-9]{1,2},\s[0-9]{4})", #date
            r"(?:Nov(?:\.|ember)\s[0-9]{1,2},\s[0-9]{4})", #date
            r"(?:Dec(?:\.|ember)\s[0-9]{1,2},\s[0-9]{4})", #date
            r"(?:Fig\.\s[0-9]+)" # Fig. 2
            # r"(?:\S)" # Everything else that's not whitespace
        )
big_regex_needed="|".join(regexes_needed)
compiled_regex_needed = re.compile(big_regex_needed, re.VERBOSE | re.I | re.UNICODE)

company = [
        'Corp.', 'Inc.', 'Ltd.', 'Co.', 
        'CORP.', 'INC.', 'LTD.', 'CO.', 'LLC.', 
        'L.L.C', 'L.L.P.', 'L.P.']
like_name = r"(?:(?:.*\s)?[A-Z][a-z_]+\.$)"
verdict_dict = [
    "affirmed", 
    "reversed", 
    "vacated", 
    "remanded",  
]
months = [
    "Jan", "Feb", "Mar", "Apr",
    "May", "Jun", "Jul", "Aug", 
    "Sept", "Oct", "Nov", "Dec"
]

def read_raw_text(file_id):
    file_name = raw_file_list[file_id]
    stored, queue = [], []
    synopsis_found, verdict_found, attorneys_found, main_body = \
        False, False, False, False
    
    with open(os.path.join(raw_data_dir, file_name), "r", encoding="cp950") as f:
        for line in f:
            if line.strip() == "Synopsis":
                synopsis_found = True
                continue
            if all([text.lower() in verdict_dict for text in line.strip().split(" and ")]):
                verdict_found = True
                continue
            if line.strip().lower() == "attorneys and law firms":
                attorneys_found = True
                continue
            if line.strip().lower() == "end of document":
                break

            if (not main_body) and \
                (verdict_found or attorneys_found):
                if queue[-3:] == [" \n", "\n", "\n"]:
                    main_body = True
                    queue = []
                else:
                    queue.append(line)
                    continue
            
            if (synopsis_found and not (verdict_found or attorneys_found))\
                or main_body:

                # remain only useful stuffs
                if main_body:
                    queue.append(line)
                if queue[-3:] == [" \n","\n", "\n"]:
                    break
                
                splitted_line = line.strip().split()
                if len(splitted_line) <= 5 \
                    or all([not text.isalnum() for text in splitted_line]):
                    continue
                stored.append(line)
        
    return stored



def add_punct(raw_text):
    bad_line = re.compile(r"(?:[A-Za-z0-9]\n)")
    matched = bad_line.findall(raw_text)
    for match in matched:
        raw_text = raw_text.replace(match, match.strip("\n") + "." + "\n")
    return raw_text


def truncate_easy(raw_text):
    sp = "\n \n\n\n"
    part_list = re.split(sp, raw_text)[:]
    if re.match(r"^\*[0-9]+", part_list[2]):
        return sp.join(part_list[:3])
    else:
        return sp.join(part_list[:2])


def merge_numbers(raw_text):
    # Before sent_tokenize the paragraphs, 
    # merge some multi-word numbers (e.g. case No., company abbr. ) into
    # one token so that they won't be separated into multiple sentences
    # at the ensueing sent_tokenize step.
    # No. 1234 => No._1234
    # Fig. 2 => Fig. 1
    matched_tokens = compiled_regex_needed.findall(raw_text)
    # if matched_tokens:
    #     for token in matched_tokens:
    #         raw_text = raw_text.replace(
    #             token, "BIG_TOKEN__" + "_".join(token.split()) + "__BIG_TOKEN" + " ")
    raw_text = raw_text.replace("(r)", " THE-R")
    return raw_text


def merge_mwe(raw_text):
    """ 
       Acknowledgement: anlp19/13.mwe/JustesonKatz95 
    """
    nlp = spacy.load('en_core_web_sm', disable=['ner,parser'])
    nlp.remove_pipe('ner')
    nlp.remove_pipe('parser')

    def getTokens(raw_text):
        """ Read the first 1000 lines of an input file """
        tokens=[]
        # with open(filename, encoding='utf-8') as file:
        # with open(filename, encoding='cp950') as file:
        lines = raw_text.split("\n")
        for idx, line in enumerate(lines):
            processed = nlp(line)
            # if "BIG_TOKEN" in line:
            #     print([t.text for t in processed])
            tokens.extend(processed)
            if idx > 1000:
                break
        return tokens

    tokens = getTokens(raw_text)
    words=[x.text for x in tokens]
    adjectives=set(["JJ", "JJR", "JJS"])
    nouns=set(["NN", "NNS", "NNP", "NNPS"])

    taglist=[]
    for x in tokens:
        if x.tag_ in adjectives:
            taglist.append("ADJ")
        elif x.tag_ in nouns:
            taglist.append("NOUN")
        elif x.tag == "IN":
            taglist.append("PREP")
        else:
            taglist.append("O")
                    
    tags=' '.join(taglist)

    def getChar2TokenMap(tags):
        """ 
        We'll search over the postag sequence, so we need to get the token ID for any
        character to be able to match the word token. 
        """
        ws=re.compile(" ")
        char2token={}
        lastStart=0
        for idx, m in enumerate(ws.finditer(tags)):
            char2token[lastStart]=idx
            lastStart=m.start()+1
        return char2token

    def getToken(tokenId, char2token):
        """ Find the token ID for given character in the POS sequence """
        while(tokenId > 0):
            if tokenId in char2token:
                return char2token[tokenId]
            tokenId-=1
        return None

    char2token = getChar2TokenMap(tags)
    p = re.compile("(((ADJ|NOUN) )+|((ADJ|NOUN) )*(NOUN PREP )((ADJ|NOUN) )*)NOUN")
    mweCount=Counter()

    for m in p.finditer(tags):
        startToken=getToken(m.start(),char2token)
        endToken=getToken(m.end(),char2token)
        to_join = words[startToken:endToken+1]
        # if any(["BIG_TOKEN" in w for w in to_join]):
        #     print(to_join)
        mwe=' '.join(to_join)
        mweCount[mwe]+=1

    # for k,v in mweCount.most_common(100):
    #     print(k,v)
    
    my_mwe = [k for (k,v) in mweCount.most_common(1000) if v > 1]
    
    def replaceMWE(text, mweList):
        """ 
            Replace all instances of MWEs in text with single token.
            MWEs are ranked from longest to shortest so that longest replacements are made first (e.g.,
            "New York City" is matched first before "New York")
        """
        sorted_by_length = sorted(mweList, key=len, reverse=True)
        for mwe in sorted_by_length:
            text=re.sub(re.escape(mwe), re.sub(" ", "_", mwe), text)
        return text
        
    processedText = replaceMWE(raw_text, my_mwe)
    return processedText


def sent_tokenize_paragraph(p):
    sent_tokenized_p = sent_tokenize(p)
    sent_tokenized_paragraph = []
    for sent in sent_tokenized_p:
        try:
            splitted = sent.split()

            # § case num
            if sent.startswith("§"):
                sent_tokenized_paragraph[-1] += sent
            
            # at ?
            elif sent.startswith("at"):
                sent_tokenized_paragraph[-1] += sent
            
            # Corp. Inc. Ltd. Co.
            elif len(splitted) > 1 and splitted[1] in company \
                and re.match(like_name, sent_tokenized_paragraph[-1]):
                sent_tokenized_paragraph[-1] += (" " + sent)
            
            elif any([sent.startswith(sfx) for sfx in company]) \
                and re.match(like_name, sent_tokenized_paragraph[-1]):
                sent_tokenized_paragraph[-1] += (" " + sent)
                try:
                    if re.match(like_name, sent_tokenized_paragraph[-2]) \
                        and sent_tokenized_paragraph[-1].split()[1] in company:
                        sent_tokenized_paragraph[-2] += (" " + sent_tokenized_paragraph.pop())
                except IndexError:
                    pass

            else:
                sent_tokenized_paragraph.append(sent) 

        except IndexError:
            sent_tokenized_paragraph.append(sent)  
    
    return sent_tokenized_paragraph


def word_tokenize_sent_tokenized_paragraph(sent_tokenized_paragraph):
    tokenized_paragragh = []
    good_sent = True
    for id_sent, sent in enumerate(sent_tokenized_paragraph):
        if not good_sent:
            sent = sent_tokenized_paragraph[id_sent - 1] + sent
            good_sent = True
        tokenized_s = word_tokenize(sent)
        tokenized_sent = []
        idx = 0
        while idx < len(tokenized_s):
            if tokenized_s[idx].startswith("BIG_TOKEN__"):
                if tokenized_s[idx].endswith("__BIG_TOKEN"):
                    delimeter = " " if any([m in tokenized_s[idx] for m in months]) else "_"
                    tokenized_sent.append(delimeter.join(tokenized_s[idx][11:-11].split("_")))
                    idx += 1
                else:
                    token = tokenized_s[idx]
                    while True:
                        idx += 1
                        try:
                            token += tokenized_s[idx]
                        except IndexError:
                            good_sent = False
                            break
                        if token.endswith("__BIG_TOKEN"):
                            delimeter = " " if any([m in token for m in months]) else "_"
                            tokenized_sent.append(delimeter.join(token[11:-11].split("_")))
                            idx += 1
                            break
            else:
                tokenized_sent.append(tokenized_s[idx])
                idx += 1
        if not good_sent:
            continue
        tokenized_paragragh.append(tokenized_sent)

    return tokenized_paragragh


def remove_punct(raw_text):
    # pucnts = [",", ".", "|"]
    pucnts = [
        ",", ".", "|", "`", "'", '"', 
        "(", ")", ":", "-", "_", "[", 
        "]", "{", "}"
        ]
    for pucnt in pucnts:
        to_sub = " " + pucnt
        raw_text = raw_text.replace(to_sub, "")
    return raw_text


def pipeline(file_id):
    file_name = raw_file_list[file_id]
    print(file_name)

    paragraphs = read_raw_text(file_id)
    paragraphs = [merge_numbers(p) for p in paragraphs]

    nlp = spacy.load("en_core_web_sm")
    suffixes = nlp.Defaults.suffixes + (r"Labs?\.$", r"Pharm\.?$", r"THE-R")
    suffix_regex = spacy.util.compile_suffix_regex(suffixes)
    nlp.tokenizer.suffix_search = suffix_regex.search

    tokenized_paragraphs = [[[token for token in sent] for sent in nlp(p).sents if len(sent) > 5] for p in paragraphs]
    
    # tokenized_paragraphs = []
    # for p in paragraphs:
    #     sent_tokenized_paragraph = sent_tokenize_paragraph(p) # sent tokenize the paragraph
    #     tokenized_paragragh = word_tokenize_sent_tokenized_paragraph(sent_tokenized_paragraph) # word tokenize the paragraph
    #     tokenized_paragraphs.append(tokenized_paragragh)
        # tokenized_paragraphs.append(sent_tokenized_paragraph)

    paras = tokenized_paragraphs # tokenized paragraphs
    raw_text = ""

    for para in paras:
        for sent in para:
            # raw_text += " ".join(sent)
            raw_text += " ".join([token.text for token in sent])
            raw_text += "\n"
        raw_text += "\n"

    # raw_text = merge_mwe(raw_text)
    # raw_text = remove_punct(raw_text)

    # raw_text = raw_text.lower()

    with open(preprocessed_data_dir + "\\" + file_name, "w", encoding='cp950') as f:
        f.write(raw_text)

raw_text = None


if __name__ == "__main__":
    # pipeline(5)
    for i in range(len(raw_file_list)):
        pipeline(i)
    # pass
