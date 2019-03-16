from flask import Flask, request, current_app
import logging, os, json
import torch
from pytorch_pretrained_bert import GPT2Tokenizer, GPT2Model, GPT2LMHeadModel

app = Flask(__name__)


USE_CUDA = os.environ.get('USE_CUDA', False) == 1

@app.route('/')
def hello():
    return "Hello! This is the GPT LangModel endpoint"

@app.route("/api/get_log_prob", methods=['POST'])
def get_log_prob():
    args = request.get_json()

    questions = args['queries']


    # Encode some inputs
    # text_1 = "Who was Jim Henson ?"
    

    
    
    if len(questions) > 32:
        log_probs = []
        for b in range(len(questions)//32 + 1):
            start_ix = b * 32
            end_ix = min(len(questions), (b + 1) * 32)
            log_probs.extend(get_seq_log_prob(current_app.model, get_batch(questions[start_ix:end_ix])))
    else:
        log_probs = get_seq_log_prob(current_app.model, get_batch(questions))

    resp = {'status': 'OK',
            'results': [{'log_probs': str(log_probs[i])} for i in range(len(log_probs))]}
    return json.dumps(resp)

def get_batch(str_in):
    tok_unpadded = [current_app.tokenizer.encode(x) for x in str_in]
    max_len = max([len(x) for x in tok_unpadded])
    tok_batch = [x + [0 for i in range(max_len - len(x))] for x in tok_unpadded]
    mask_batch = [[1 for i in range(len(x))] + [0 for i in range(max_len - len(x))] for x in tok_unpadded]
    
    return tok_batch, mask_batch

def get_seq_log_prob(model, batch):
    
    tokens, mask = batch
    
    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor(tokens)
    mask_tensor = torch.tensor(mask, dtype=torch.float)
    
    if USE_CUDA:
        tokens_tensor = tokens_tensor.to('cuda')

    # Predict all tokens
    with torch.no_grad():
        predictions, _ = model(tokens_tensor)
    all_probs = torch.softmax(predictions, -1)

    
    
    # print(all_probs.size(), all_probs)
    # print(tokens_tensor.unsqueeze(-1).size(), tokens_tensor.unsqueeze(-1))
    
    probs = torch.gather(all_probs, 2, tokens_tensor.unsqueeze(-1)).squeeze(-1)

    log_probs = torch.log(probs)
    nll = -1 * torch.mean(log_probs * mask_tensor, -1)
    return nll.tolist()



def init():
    print('Spinning up LangModel (GPT) service')



    # OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
    
    logging.basicConfig(level=logging.INFO)

    # Load pre-trained model tokenizer (vocabulary)
    app.tokenizer = GPT2Tokenizer.from_pretrained('gpt2', cache_dir='../models')

    # Load pre-trained model (weights)
    app.model = GPT2LMHeadModel.from_pretrained('gpt2', cache_dir='../models')
    app.model.eval()

    # If you have a GPU, put everything on cuda
    if USE_CUDA:
        app.model.to('cuda')


    
    

if __name__ == '__main__':
    init()
    with app.app_context():
        app.run(host="0.0.0.0", port=5006, processes=1)