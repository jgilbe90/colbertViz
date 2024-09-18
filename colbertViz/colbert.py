from typing import List
import os
import re
import torch
from colbert.modeling.hf_colbert import class_factory
from colbert.infra import ColBERTConfig
from colbert.modeling.colbert import ColBERT
from colbert.modeling.tokenization import QueryTokenizer, DocTokenizer
from html2image import Html2Image

try:
    from IPython.display import display, HTML
    NOTEBOOK = True
except:
    NOTEBOOK = False

class colbertViz:
    def __init__(self, 
                nbits: int = 2, 
                doc_maxlen: int = 512, 
                checkpoint: str = "colbert-ir/colbertv2.0", 
                output_image_size: tuple = (700, 500),
                output_image_prefix: str = "colbertViz",
                output_image_directory: str = '.'):
        config = ColBERTConfig(doc_maxlen=doc_maxlen, nbits=nbits, nway=1, checkpoint="colbert-ir/colbertv2.0")
        self.model = ColBERT(colbert_config=config)
        self.qtokenizer = QueryTokenizer(config)
        self.dtokenizer = DocTokenizer(config)
        self.notebook = NOTEBOOK
        if not NOTEBOOK:
            self.save_prefix = output_image_prefix
            self.save_directory = output_image_directory
            self.imageWriter = Html2Image(size=output_image_size)

    def viz(self, query: str, docs: List[str]):
        qtoks = self.qtokenizer.tensorize([query])
        dtoks = self.dtokenizer.tensorize(docs)

        #colbert forward implementation
        Q, D = qtoks, dtoks

        Q = self.model.query(*Q)
        #D_mask is False for any punctuation or special tokens (to ignore later)
        D, D_mask = self.model.doc(*D, keep_dims='return_mask')

        #only used if more than 1 query or doc
        Q_duplicated = Q.repeat_interleave(self.model.colbert_config.nway, dim=0).contiguous()

        Q, D_padded, D_mask = Q_duplicated, D, D_mask

        #calculate cosine similarity for each term of doc and query
        #cosine similarity = dot product for normalized vector
        scores = D_padded @ Q.to(dtype=D_padded.dtype).permute(0, 2, 1)

        scores_padded, D_mask = scores, D_mask
        #Set all punctuation tensors from doc to low value so they dont impact max sim
        D_padding = ~D_mask.view(scores_padded.size(0), scores_padded.size(1)).bool()
        scores_padded[D_padding] = -9999
        scores = scores_padded.max(1).values

        #colbert is sum of maxs on dim 0 (query terms)
        #so max of dim 1 (doc terms) estimates their importance
        #ignore any masked/ special tokens in query
        doc_scores, query_term_idx = scores_padded[0,2:-1,2:qtoks[1].sum()-1].max(dim=1)

        '''
        #min max scale tensor (ignoring masked/ zero scores)
        doc_scores = torch.clamp(doc_scores, 0).float()
        zero_mask = doc_scores > 0
        min_val, max_val = doc_scores[zero_mask].min(), doc_scores[zero_mask].max()
        doc_scores[zero_mask] = (doc_scores[zero_mask] - min_val) / (max_val - min_val)
        '''
        
        #get colberts tokenizer
        HF_ColBERT = class_factory(self.model.colbert_config.checkpoint)
        tokenizer = HF_ColBERT.raw_tokenizer_from_pretrained(self.model.colbert_config.checkpoint)
        
        #decode document
        top_doc_toks, top_query_toks = [], []
        for dtoken, qtoken in zip(dtoks[0][0], qtoks[0][0]):
            top_doc_toks.append(tokenizer.decode(dtoken))
            top_query_toks.append(tokenizer.decode(qtoken))

        tokens, scores = align_doc_tokens_to_terms(top_doc_toks[2:-1], doc_scores)
        query_terms, query_term_idx = align_query_tokens_to_terms(top_query_toks, query_term_idx)
        html_str = build_html(tokens, scores, query_terms, query_term_idx)

        if self.notebook:
            display(HTML(html_str))
        else:
            existing_ids = [re.match(f"{self.save_prefix}(\\d+)\.png", file) for file in os.listdir(self.save_directory)]
            existing_ids = sorted([int(m.group(1)) for m in existing_ids if m])
            save_id = '1' if len(existing_ids) < 1 else str(existing_ids[-1]+1)
            write_name = f'{self.save_prefix}{save_id}.png'
            self.imageWriter.screenshot(html_str=html_str, save_as=write_name)
            return write_name

#helper function to find max score of any words represented by multiple tokens
def align_query_tokens_to_terms(top_query_toks, query_term_idx):
    combined_querys = []
    query_mapping = {}
    current_query = -1
    current_token = ''
    for idx, token in enumerate(top_query_toks):
        if not token.startswith('##'):
            if current_token:
                combined_querys.append(current_token)
                current_token = ''
            current_query += 1
            current_token = token
        else:
            current_token += token[2:]

        query_mapping[idx] = current_query

    if current_token not in combined_querys[-1]:
        combined_querys.append(current_token)

    return combined_querys, list(map(lambda term: query_mapping[term],query_term_idx.tolist()))

#helper function to find max score of any words represented by multiple tokens
def align_doc_tokens_to_terms(top_doc_toks, doc_scores):
    combined_tokens = []
    combined_scores = []
    current_token = ''
    current_score = 0

    for token, score in zip(top_doc_toks, doc_scores):
        if token.startswith('##'):
            # Remove '##' and append to the current token
            current_token += token[2:]
            current_score = max(current_score, score)
        else:
            # If there is a current token, add it to the list
            if current_token:
                combined_tokens.append(current_token)
                combined_scores.append(current_score)
                current_token, current_score = '', 0
            # Add the new token to the list
            current_token = token
            current_score = score

    # Add the last token if there's any
    if current_token:
        combined_tokens.append(current_token)
        combined_scores.append(current_score)

    return combined_tokens, combined_scores

#helper function to convert doc terms and relevancy scores to highlighted HTML to display
def build_html(combined_tokens, combined_scores, query_terms, query_term_idx):
    html_str = """<style>
    .tooltip {
    position: relative;
    display: inline-block;
    }

    .tooltip .tooltiptext {
    visibility: hidden;
    max-width: 300px; /* Set a maximum width */
    background-color: black;
    color: #fff;
    text-align: left;
    border-radius: 6px;
    padding: 5px 10px;
    position: absolute;
    z-index: 1;
    top: 100%;
    left: 50%;
    transform: translateX(-50%); /* Center the tooltip */
    white-space: nowrap; /* Prevent text from wrapping */
    overflow-y: hidden;
    text-overflow: ellipsis; /* Optional: Adds '...' if the text is too long */
    }

    .tooltip:hover .tooltiptext {
    visibility: visible;
    opacity: 1;
    }

    .tooltip .tooltiptext::after {
    content: '';
    position: absolute;
    bottom: 100%; /* Position the arrow above the tooltip */
    left: 50%;
    transform: translateX(-50%);
    border-width: 5px;
    border-style: solid;
    border-color: transparent transparent black transparent; /* Arrow pointing down */
    z-index: 1;
    }

    :not(.tooltip) {
    font-size: 18px;
    line-height: 2;
    word-spacing: 2px;
    }

    </style>"""
    for token, score, query_idx in zip(combined_tokens, combined_scores, query_term_idx):
        if score > 0.5:
            display_score = round(score.item(),2)
            to_add = f'<div class="tooltip"><span class="tooltiptext"><b>Relevancy:</b> {display_score}<br><b>Query Term:</b> {query_terms[query_idx]}</span><span style="background-color: rgba(255, 0, 0, {display_score}); border-radius: 0.5rem; padding: 5px; color: black;"><b>{token}</b></span></div> '
        else:
            to_add = f"{token} "
        html_str += to_add

    return html_str
