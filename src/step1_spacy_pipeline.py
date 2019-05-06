import os
import re
import pickle
from collections import Counter
import spacy
from spacy.matcher import Matcher
from spacy.tokens import Token
from utils import raw_file_list, raw_data_dir, spacy_docs_dir, \
    preprocessed_tokens_dir, preprocessed_sents_dir, save_stuff
from spacy.symbols import ORTH, LEMMA, POS, TAG


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
        'Corporation', 'corporation', "Company", "company", 
        'Limited Partnership', 
        'L.L.C', 'llc' 'L.L.P.', 'L.P.', 
        'Laboratories', 'Labs', 'Lab.', 
        'Pharm.', 'Pharma.']
org_abbr_patterns=(
            r"(?:[Cc]orp\.(?:'s)?\s\(.+?\))",
            r"(?:[Cc]orporation(?:'s)?\s\(.+?\))",
            r"(?:[Ii]nc\.(?:'s)?\s\(.+?\))", 
            r"(?:[Ll]td\.(?:'s)?\s\(.+?\))",
            r"(?:[Cc]o\.(?:'s)?\s\(.+?\))",
            r"(?:[Cc]ompany(?:'s)?\s\(.+?\))",
            r"(?:LLC(?:\.)?(?:'s)?\s\(.+?\))",
            r"(?:llc(?:\.)?(?:'s)?\s\(.+?\))",
            r"(?:L.L.C.(?:\.)?(?:'s)?\s\(.+?\))",
            r"(?:LLP(?:\.)?(?:'s)?\s\(.+?\))",
            r"(?:llp(?:\.)?(?:'s)?\s\(.+?\))",
            r"(?:L.L.P(?:\.)?(?:'s)?\s\(.+?\))",
            r"(?:LP(?:\.)?(?:'s)?\s\(.+?\))",
            r"(?:lp(?:\.)?(?:'s)?\s\(.+?\))",
            r"(?:L.L.P(?:\.)?(?:'s)?\s\(.+?\))",
            r"(?:Limited\sParternership(?:'s)?\s\(.+?\))",
            r"(?:[Ll]ab\.(?:'s)?\s\(.+?\))", 
            r"(?:Laboratories(?:'s)?\s\(.+?\))", 
            r"(?:[Pp]harma?\.(?:'s)?\s\(.+?\))", 
            r"(?:Pharmacy(?:'s)?\s\(.+?\))", 
            r"(?:\(collectively.+?\))", 
        )
big_org_abbr_patterns="|".join(org_abbr_patterns)
compiled_org_abbr_patterns = re.compile(big_org_abbr_patterns, re.VERBOSE | re.I | re.UNICODE)

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


# create spacy pipeline

# modify tokenizer
# nlp = spacy.load("en_core_web_lg")
nlp = spacy.load("en_core_web_sm", disable=['ner'])
# nlp.remove_pipe('ner')
special_case = [{ORTH: u"THE-R", LEMMA: u"(r)", TAG: u"SYM"}]
nlp.tokenizer.add_special_case(u"THE-R", special_case)
special_case = [{ORTH: u"THE-R", LEMMA: u"(r)", TAG: u"SYM"}, {ORTH: u".", LEMMA: u".", TAG: u"."}]
nlp.tokenizer.add_special_case(u"THE-R.", special_case)
special_case = [{ORTH: u"THE-R", LEMMA: u"(r)", TAG: u"SYM"}, {ORTH: u",", LEMMA: u",", TAG: u","}]
nlp.tokenizer.add_special_case(u"THE-R,", special_case)
special_case = [{ORTH: u"No.", LEMMA: u"No.", TAG: u"SYM"}]
nlp.tokenizer.add_special_case(u"No.", special_case)
special_case = [{ORTH: u"Nos.", LEMMA: u"No.", TAG: u"SYM"}]
nlp.tokenizer.add_special_case(u"Nos.", special_case)
special_case = [{ORTH: u'"', TAG: u"punct"}, {ORTH: u')', LEMMA: u'-RRB-', TAG: u"punct"}]
nlp.tokenizer.add_special_case(u'")', special_case)
special_case = [{ORTH: u"α-Gal-A", LEMMA: u"α-Gal-A", TAG: u"NNP"}, {ORTH: u'.', LEMMA: u'.', TAG: u"punct"}]
nlp.tokenizer.add_special_case(u'α-Gal-A.', special_case)
special_case = [{ORTH: u"v.", LEMMA: u"v.", TAG: u"CC"}]
nlp.tokenizer.add_special_case(u'v.', special_case)
special_case = [{ORTH: u"kD", LEMMA: u"kD", TAG: u"SYM"}]
nlp.tokenizer.add_special_case(u'kD', special_case)
suffixes = nlp.Defaults.suffixes + (r"Labs?\.$", r"Inc\.$", r"Pharm\.?$", r"THE-R", r"col.", r"Col.")
suffix_regex = spacy.util.compile_suffix_regex(suffixes)
nlp.tokenizer.suffix_search = suffix_regex.search

def set_custom_boundaries(doc):
    para_mismatch = False
    sqbrackets_mismatch = False
    brace_mismatch = False
    quote_mismatch = False
    for token in doc[:-1]:
        if doc[token.i-1].lemma_ != ".":
            doc[token.i].is_sent_start = False

        if token.lemma_ == "(r)":
            if doc[token.i-2].tag_ != ".":
                doc[token.i-1].is_sent_start = False
            doc[token.i].is_sent_start = False
        elif token.text == "kD" and not doc[token.i+1].is_punct:
            doc[token.i+1].is_sent_start = False
            if doc[token.i-1].tag_ == "CD":
                doc[token.i-1].is_sent_start = False
        elif token.text in ["BACKGROUND", "Background", "INVENTION"]:
            doc[token.i+2].is_sent_start = True
        elif token.text in company:
            doc[token.i].is_sent_start = False
        elif token.text == ",":
            doc[token.i].is_sent_start = False
        elif token.lemma_ == ":":
            if doc[token.i+1].text == "\n ":
                doc[token.i+2].is_sent_start = True
            else:
                doc[token.i+1].is_sent_start = True
        elif token.text == "\n ":
            doc[token.i+1].is_sent_start = True
        
        if token.text.count("{") == token.text.count("}") + 1:
            brace_mismatch = True
            continue
        if token.text.count("[") == token.text.count("]") + 1:
            sqbrackets_mismatch = True
            continue
        if token.text.count("(") == token.text.count(")") + 1:
            para_mismatch = True
            if doc[token.i-1].lemma_ != ".":
                doc[token.i].is_sent_start = False
            continue
        if token.text.count('"') % 2 != 0 and not quote_mismatch:
            quote_mismatch = True
            continue
        if token.text.count("{") + 1 == token.text.count("}") and brace_mismatch:
            brace_mismatch = False
            doc[token.i].is_sent_start = False
            continue
        if token.text.count("[") + 1 == token.text.count("]") and sqbrackets_mismatch:
            sqbrackets_mismatch = False
            doc[token.i].is_sent_start = False
            continue
        if token.text.count("(") + 1 == token.text.count(")") and para_mismatch:
            para_mismatch = False
            doc[token.i].is_sent_start = False
            if doc[token.i+1].lemma_ != ".":
                doc[token.i+1].is_sent_start = False
            continue
        if token.text.count('"') % 2 != 0 and quote_mismatch:
            quote_mismatch = False
            doc[token.i].is_sent_start = False
            continue

        if para_mismatch or sqbrackets_mismatch or quote_mismatch or brace_mismatch:
            doc[token.i].is_sent_start = False

    return doc

nlp.add_pipe(set_custom_boundaries, before="parser")


# merge some specific tokens like case numbers 
# # Acknowledgement: https://spacy.io/usage/rule-based-matching
class Merger(object):
    def __init__(self, nlp):
        # Register a new token extension to flag bad HTML
        Token.set_extension("col_num", default=False, force=True)
        self.matcher = Matcher(nlp.vocab)
        self.matcher.add(
            "col_num_with_comma", None,
            [{"TEXT": {"REGEX": r"[Cc]ol\."}}, {"TEXT": {"REGEX": r"[0-9]+"}}, 
            {"TEXT": ","}, {"TEXT": {"REGEX": r"l+\."}}, 
            {"TEXT": {"REGEX": r"[0-9\-]+"}}]
        )
        self.matcher.add(
            "col_num", None,
            [{"TEXT": {"REGEX": r"[Cc]ol\."}}, {"TEXT": {"REGEX": r"[0-9]+"}}, 
            {"TEXT": {"REGEX": r"l+\."}}, {"TEXT": {"REGEX": r"[0-9\-]+"}}]
        )

    def __call__(self, doc):
        # This method is invoked when the component is called on a Doc
        matches = self.matcher(doc)
        spans = []  # Collect the matched spans here
        for match_id, start, end in matches:
            spans.append(doc[start:end])
        with doc.retokenize() as retokenizer:
            for span in spans:
                retokenizer.merge(span)
                for token in span:
                    token._.col_num = True  # Mark token as col_num
        return doc

merger = Merger(nlp)
nlp.add_pipe(merger, last=True)  # Add component to the pipeline

# merge noun-phrases
# merge_nps = nlp.create_pipe("merge_noun_chunks")
# nlp.add_pipe(merge_nps)


def read_raw_text(file_id, truncate=False):
    file_name = raw_file_list[file_id]
    stored, queue = "", []
    synopsis_found, verdict_found, attorneys_found, judge_found, main_body = \
        False, False, False, False, False
    
    with open(os.path.join(raw_data_dir, file_name), "r", encoding="cp950") as f:
        for line in f:
            if line.strip() == "Synopsis":
                synopsis_found = True
                continue
            if all([text.lower() in verdict_dict for text in line.strip().strip(".").split(" and ")]):
                verdict_found = True
                continue
            if line.strip().lower() == "attorneys and law firms":
                attorneys_found = True
                continue
            if len(line.strip().split()) >= 1 and line.strip().split()[-1] == "Judge.":
                judge_found = True
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
            
            if (synopsis_found and not (verdict_found or attorneys_found))\
                or judge_found \
                or main_body:
                # remain only useful stuffs
                if main_body:
                    queue.append(line)
                if truncate and queue[-3:] == [" \n","\n", "\n"]:
                    break
                
                splitted_line = line.strip().split()
                # if len(splitted_line) <= 5 \
                #     or all([not text.isalnum() for text in splitted_line]):
                if (len(splitted_line) == 1 and all(c.upper() == c for c in [splitted_line[0]]))\
                    or all([not text.isalnum() for text in splitted_line]):
                    continue

                stored += line + " "
        
    # print(stored)
    return stored


def find_org_abbr(raw_text):
    abbrs = []

    # find abbr for corp in title
    matches = re.findall(compiled_org_abbr_patterns, raw_text)
    for m in matches:
        if '"' in m:
            m = m.split('"')
            abbr = m[1]
        else:
            m = m.split("(")[1]
            m = m.strip(")")
            abbr = m
        abbr = " ".join(filter(lambda s: not (s.isalnum() and s.lower() == s), abbr.split()))
        abbrs.append(abbr)
    
    # find other orgs
    vs_pattern = r"(?:(?:[A-Z][A-Za-z0-9\.\-]*,? )+vs?\.(?: [A-Z][A-Za-z0-9\.\-]*,?)+)"
    vs_matches = re.findall(vs_pattern, raw_text)
    for m in vs_matches:
        parties = m.split(" v. ")
        for party in parties:
            if not re.findall(r"(?:(?:I)|(?:II)|(?:III)|(?:IV)|(?:V)|(?:VI)|(?:VII)|(?:VIII)|(?:IX)|(?:X))", party):
                abbrs.append(party.split()[0])

    return set(abbrs)


def find_per_names(raw_text):
    abbrs= []

    title_pattern = r"(?:[MD]r\.\s[A-Z][a-z]*)"
    name_pattern = r"(?:[A-Z][a-z]*\s(?:[A-Z]\.)+\s[A-Z][a-z]*)"
    
    titles = re.findall(title_pattern, raw_text)
    names = re.findall(name_pattern, raw_text)
    # print(titles)
    # print(names)

    for title in titles:
        abbrs.append(title.split()[1])

    for name in names:
        abbrs.extend(name.split())

    return set(abbrs)


def preprocess(raw_text):
    # remove annotation like "*1234"
    raw_text = re.sub(r"\s\*\d+\s", " ", raw_text)

    # substitute '."' to '".' to make sentence segmetation correct
    misplaced_quote_period = re.findall(r'(?:[A-Za-z0-9]+\.")', raw_text)
    for m in misplaced_quote_period:
        raw_text = raw_text.replace(m, m[:-2] + '".')
    
    # make "(r)" a special token
    raw_text = raw_text.replace("(r)", " THE-R")

    # correct typos
    # (e.g. "[T]hanos" to "Thanos")
    # (e.g. "Than[os]" to "Thanos")
    rdnt_sq_brckts = re.findall(r"(?:(?:[^\s\d\[\]]+\[\w{0,2}\][^\s\d\[\]]+)|(?:\s\[\w{0,2}\][^\s\d\[\]]+)|(?:[^\s\d\[\]]+\[\w{0,2}\]\s))", raw_text)
    for m in rdnt_sq_brckts:
        substitution = m.replace("[", "")
        substitution = substitution.replace("]", "")
        raw_text = raw_text.replace(m, substitution)

    return raw_text


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
    global org_and_per_names
    file_name = raw_file_list[file_id]
    print(file_name)

    # find the org and per names in the document
    with open(os.path.join(raw_data_dir, file_name), "r", encoding="cp950") as f:
        org_and_per_name_doc = file_name[3:].partition(".txt")[0].split(" v ")
        first_words = [total.split()[0] for total in org_and_per_name_doc]
        org_and_per_name_doc.extend(first_words)
        r_text = f.read()
        org_and_per_name_doc.extend(find_org_abbr(r_text))
        org_and_per_name_doc.extend(find_per_names(r_text))
        org_and_per_name_doc = set(org_and_per_name_doc)
        # print(org_and_per_name_doc)
        org_and_per_names.append(org_and_per_name_doc)

    raw_text = read_raw_text(file_id)
    raw_text = preprocess(raw_text)
    doc = nlp(raw_text)

    # write as tokens
    with open(os.path.join(preprocessed_tokens_dir, file_name), "w", encoding='cp950') as f:
        for sent in doc.sents:
            for token in sent:
                f.write("\t".join([token.lemma_, str(token.tag_), str(token.dep_)]))
                f.write("\n")
            f.write("\n")

    # write as sents
    with open(os.path.join(preprocessed_sents_dir, file_name), "w", encoding='cp950') as f:
        for sent in doc.sents:
            f.write("\t" + sent.text.replace("\n", ""))
            f.write("\n")

    # save the spacy doc
    with open(os.path.join(spacy_docs_dir, file_name + ".pk"), "wb") as f:
        pickle.dump(doc, f)

if __name__ == "__main__":
    org_and_per_names = []
    #################################################################################################################
    # BE CAREFUL: running pipeline(i) below will remove the manually labeled "key-sentence" labels in the file.     #
    # You may have to label the sentences again in the each file in the directory "../ivan_data/preprocessed/sents" #
    #################################################################################################################
    
    # pipeline(318)
    # for i in range(len(raw_file_list)):
    #     pipeline(i)
    # save_stuff(org_and_per_names, "org_and_per_names")
