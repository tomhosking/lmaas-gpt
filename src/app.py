from flask import Flask, request, current_app


app = Flask(__name__)


@app.route('/')
def hello():
    return "Hello! This is the GPT LangModel endpoint"

@app.route("/api/get_log_prob", methods=['POST'])
def get_log_prob():
    args = request.get_json()

    questions = args['queries']


    # Encode some inputs
    text_1 = "Who was Jim Henson ?"
    indexed_tokens_1 = tokenizer.encode(text_1)

    log_probs = get_seq_log_prob(current_app.model, indexed_tokens_1)
    
    # if len(questions) > 32:
    #     log_probs = []
    #     for b in range(len(questions)//32 + 1):
    #         start_ix = b * 32
    #         end_ix = min(len(questions), (b + 1) * 32)
    #         log_probs.extend(current_app.generator.get_seq_perplexity(questions[start_ix:end_ix]))
    # else:
    #     log_probs = current_app.generator.get_seq_perplexity(questions)

    resp = {'status': 'OK',
            'results': [{'log_probs': str(log_probs[i])} for i in range(len(log_probs))]}
    return json.dumps(resp)

def get_seq_log_prob(model, tokens)
    

    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([tokens])
    
    tokens_tensor = tokens_tensor.to('cuda')

    # Predict all tokens
    with torch.no_grad():
        predictions, _ = model(tokens_tensor)
        
    probs = torch.gather(predictions, 2, tokens_tensor)

    log_probs = torch.log(probs)

    return torch.mean(log_probs, -1)



def init():
    print('Spinning up LangModel (GPT) service')


    import torch
    from pytorch_pretrained_bert import GPT2Tokenizer, GPT2Model, GPT2LMHeadModel

    # OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
    import logging
    logging.basicConfig(level=logging.INFO)

    # Load pre-trained model tokenizer (vocabulary)
    app.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    # Load pre-trained model (weights)
    app.model = GPT2LMHeadModel.from_pretrained('gpt2')
    app.model.eval()

    # If you have a GPU, put everything on cuda
    app.model.to('cuda')


    
    

if __name__ == '__main__':
    init()
    with app.app_context():
        app.run(host="0.0.0.0", port=5006, processes=1)