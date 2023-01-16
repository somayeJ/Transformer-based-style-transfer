import os
import time
import torch
import numpy as np
from torch import nn, optim
from torch.nn.utils import clip_grad_norm_
from utils import tensor2text, word_drop, word_dropout
from file_io import write_sent

def get_lengths(tokens, eos_idx):
    lengths = torch.cumsum(tokens == eos_idx, 1)
    lengths = (lengths == 0).long().sum(-1)
    lengths = lengths + 1 
    return lengths

def batch_preprocess(batch, pad_idx, eos_idx, reverse=False): 
    batch_pos, batch_neg = batch
    diff = batch_pos.size(1) - batch_neg.size(1)
    if diff < 0:
        pad = torch.full_like(batch_neg[:, :-diff], pad_idx)
        batch_pos = torch.cat((batch_pos, pad), 1)
    elif diff > 0:
        pad = torch.full_like(batch_pos[:, :diff], pad_idx)
        batch_neg = torch.cat((batch_neg, pad), 1)

    pos_styles = torch.ones_like(batch_pos[:, 0])
    neg_styles = torch.zeros_like(batch_neg[:, 0])

    if reverse:
        batch_pos, batch_neg = batch_neg, batch_pos
        pos_styles, neg_styles = neg_styles, pos_styles
        
    tokens = torch.cat((batch_pos, batch_neg), 0)
    lengths = get_lengths(tokens, eos_idx)
    styles = torch.cat((pos_styles, neg_styles), 0)

    return tokens, lengths, styles
        
def d_step(config, vocab, model_F, model_D, optimizer_D, batch, temperature):
    model_F.eval()
    pad_idx = vocab.stoi['<pad>']
    eos_idx = vocab.stoi['<eos>']
    vocab_size = len(vocab)
    loss_fn = nn.NLLLoss(reduction='none')

    inp_tokens, inp_lengths, raw_styles = batch_preprocess(batch, pad_idx, eos_idx)
    rev_styles = 1 - raw_styles
    batch_size = inp_tokens.size(0)

    with torch.no_grad():
        raw_gen_log_probs = model_F(
            inp_tokens, 
            None,
            inp_lengths,
            raw_styles,
            generate=True,
            differentiable_decode=True,
            temperature=temperature,
        )
        rev_gen_log_probs = model_F(
            inp_tokens,
            None,
            inp_lengths,
            rev_styles,
            generate=True,
            differentiable_decode=True,
            temperature=temperature,
        )
 
    raw_gen_soft_tokens = raw_gen_log_probs.exp()
    raw_gen_lengths = get_lengths(raw_gen_soft_tokens.argmax(-1), eos_idx)

    rev_gen_soft_tokens = rev_gen_log_probs.exp()
    rev_gen_lengths = get_lengths(rev_gen_soft_tokens.argmax(-1), eos_idx)

    raw_gen_log_probs = model_D(raw_gen_soft_tokens, raw_gen_lengths, raw_styles)
    rev_gen_log_probs = model_D(rev_gen_soft_tokens, rev_gen_lengths, rev_styles)
    gen_log_probs = torch.cat((raw_gen_log_probs, rev_gen_log_probs), 0)
    raw_gen_labels = torch.ones_like(raw_styles)
    rev_gen_labels = torch.zeros_like(rev_styles)
    gen_labels = torch.cat((raw_gen_labels, rev_gen_labels), 0)

    if config.discriminator_gold_train:
        raw_gold_log_probs = model_D(inp_tokens, inp_lengths, raw_styles)
        rev_gold_log_probs = model_D(inp_tokens, inp_lengths, rev_styles)
        gold_log_probs = torch.cat((raw_gold_log_probs, rev_gold_log_probs), 0)
        raw_gold_labels = torch.ones_like(raw_styles)
        rev_gold_labels = torch.zeros_like(rev_styles )
        gold_labels = torch.cat((raw_gold_labels, rev_gold_labels), 0)

        adv_log_probs = torch.cat((gold_log_probs, gen_log_probs), 0)
        adv_labels = torch.cat((gold_labels, gen_labels), 0)
        adv_loss = loss_fn(adv_log_probs, adv_labels)
        assert len(adv_loss.size()) == 1
        adv_loss = adv_loss.sum() / batch_size
        loss = adv_loss
    else:
        adv_loss = loss_fn(gen_log_probs,  gen_labels)
        assert len(adv_loss.size()) == 1
        batch_size_modified = batch_size * 0.5
        adv_loss = adv_loss.sum() / batch_size_modified
        loss = adv_loss

    optimizer_D.zero_grad()
    loss.backward()
    clip_grad_norm_(model_D.parameters(), 5)
    optimizer_D.step()

    model_F.train()

    return adv_loss.item()

def d_step_test(config, vocab, model_F, model_D, batch, temperature):
    model_F.eval()
    model_D.eval()
    pad_idx = vocab.stoi['<pad>']
    eos_idx = vocab.stoi['<eos>']
    vocab_size = len(vocab)
    loss_fn = nn.NLLLoss(reduction='none')

    inp_tokens, inp_lengths, raw_styles = batch_preprocess(batch, pad_idx, eos_idx)
    rev_styles = 1 - raw_styles
    batch_size = inp_tokens.size(0)

    with torch.no_grad():
        raw_gen_log_probs = model_F(
            inp_tokens, 
            None,
            inp_lengths,
            raw_styles,
            generate=True,
            differentiable_decode=True,
            temperature=temperature,
        )
        rev_gen_log_probs = model_F(
            inp_tokens,
            None,
            inp_lengths,
            rev_styles,
            generate=True,
            differentiable_decode=True,
            temperature=temperature,
        )

    raw_gen_soft_tokens = raw_gen_log_probs.exp()
    raw_gen_lengths = get_lengths(raw_gen_soft_tokens.argmax(-1), eos_idx)
    
    rev_gen_soft_tokens = rev_gen_log_probs.exp()
    rev_gen_lengths = get_lengths(rev_gen_soft_tokens.argmax(-1), eos_idx)

    with torch.no_grad():
        raw_gen_log_probs = model_D(raw_gen_soft_tokens, raw_gen_lengths, raw_styles)
        rev_gen_log_probs = model_D(rev_gen_soft_tokens, rev_gen_lengths, rev_styles)
        gen_log_probs = torch.cat((raw_gen_log_probs, rev_gen_log_probs), 0)
        raw_gen_labels = torch.ones_like(raw_styles)
        rev_gen_labels = torch.zeros_like(rev_styles)
        gen_labels = torch.cat((raw_gen_labels, rev_gen_labels), 0)

        if config.discriminator_gold_train:
            raw_gold_log_probs = model_D(inp_tokens, inp_lengths, raw_styles)
            rev_gold_log_probs = model_D(inp_tokens, inp_lengths, rev_styles)
            gold_log_probs = torch.cat((raw_gold_log_probs, rev_gold_log_probs), 0)
            raw_gold_labels = torch.ones_like(raw_styles)
            rev_gold_labels = torch.zeros_like(rev_styles)
            gold_labels = torch.cat((raw_gold_labels, rev_gold_labels), 0)

            adv_log_probs = torch.cat((gold_log_probs, gen_log_probs), 0)
            adv_labels = torch.cat((gold_labels, gen_labels), 0)
            adv_loss = loss_fn(adv_log_probs, adv_labels)
            assert len(adv_loss.size()) == 1
            adv_loss = adv_loss.sum() / batch_size
            loss = adv_loss
        else:
            adv_loss = loss_fn(gen_log_probs,  gen_labels)
            assert len(adv_loss.size()) == 1
            batch_size_modified = batch_size * 0.5
            adv_loss = adv_loss.sum() / batch_size_modified
            loss = adv_loss

    return adv_loss.item()

def f_step(config, vocab, model_F, model_D, optimizer_F, batch, temperature, drop_decay,
           cyc_rec_enable=True,  optimize_total_loss = False):
    model_D.eval()
    
    pad_idx = vocab.stoi['<pad>']
    eos_idx = vocab.stoi['<eos>']
    unk_idx = vocab.stoi['<unk>']
    vocab_size = len(vocab)
    loss_fn = nn.NLLLoss(reduction='none')

    inp_tokens, inp_lengths, raw_styles = batch_preprocess(batch, pad_idx, eos_idx)
    rev_styles = 1 - raw_styles
    batch_size = inp_tokens.size(0)
    token_mask = (inp_tokens != pad_idx).float()

    optimizer_F.zero_grad()

    noise_inp_tokens = word_dropout(
        inp_tokens,
        inp_lengths, 
        config.inp_drop_prob * drop_decay,
        vocab 
    )
    noise_inp_lengths = get_lengths(noise_inp_tokens, eos_idx)

    slf_log_probs = model_F(
        noise_inp_tokens, 
        inp_tokens, 
        noise_inp_lengths,
        raw_styles,
        generate=False,
        differentiable_decode=False,
        temperature=temperature,
    )

    slf_rec_loss = loss_fn(slf_log_probs.transpose(1, 2), inp_tokens) * token_mask
    slf_rec_loss = slf_rec_loss.sum() / batch_size
    slf_rec_loss *= config.slf_factor # slf_factor = 0.25

    gen_log_probs = model_F(
        inp_tokens,
        None,
        inp_lengths,
        rev_styles,
        generate=True,
        differentiable_decode=True,
        temperature=temperature,
    )# style-shifted seq

    gen_soft_tokens = gen_log_probs.exp()
    gen_lengths = get_lengths(gen_soft_tokens.argmax(-1), eos_idx)

    # style consistency loss
    adv_log_porbs = model_D(gen_soft_tokens, gen_lengths, rev_styles)

    adv_labels = torch.ones_like(rev_styles)# 
    adv_loss = loss_fn(adv_log_porbs, adv_labels)
    adv_loss = adv_loss.sum() / batch_size
    adv_loss *= config.adv_factor # 
    
    # cycle consistency loss
    if not cyc_rec_enable:
        if optimize_total_loss:
            (slf_rec_loss + adv_loss).backward()
        else:
            slf_rec_loss.backward()
        # update parameters
        clip_grad_norm_(model_F.parameters(), 5)
        optimizer_F.step()
        model_D.train()
        return slf_rec_loss.item(), adv_loss.item()

    cyc_log_probs = model_F(
        gen_soft_tokens,
        inp_tokens,
        gen_lengths,
        raw_styles,
        generate=False,
        differentiable_decode=False,
        temperature=temperature,
    )

    cyc_rec_loss = loss_fn(cyc_log_probs.transpose(1, 2), inp_tokens) * token_mask
    cyc_rec_loss = cyc_rec_loss.sum() / batch_size
    cyc_rec_loss *= config.cyc_factor 

    if optimize_total_loss:
            (slf_rec_loss+cyc_rec_loss + adv_loss).backward()
    else:
        (slf_rec_loss+cyc_rec_loss).backward()
        
        
    # update parameters
    clip_grad_norm_(model_F.parameters(), 5)
    optimizer_F.step()

    model_D.train()

    return slf_rec_loss.item(), cyc_rec_loss.item(), adv_loss.item()

def f_step_test(config, vocab, model_F, model_D,  batch, temperature, drop_decay,
           cyc_rec_enable=True):
    model_D.eval()
    model_F.eval()
    pad_idx = vocab.stoi['<pad>']
    eos_idx = vocab.stoi['<eos>']
    unk_idx = vocab.stoi['<unk>']
    vocab_size = len(vocab)
    loss_fn = nn.NLLLoss(reduction='none')

    inp_tokens, inp_lengths, raw_styles = batch_preprocess(batch, pad_idx, eos_idx)
    rev_styles = 1 - raw_styles
    batch_size = inp_tokens.size(0)
    token_mask = (inp_tokens != pad_idx).float()

    noise_inp_tokens = word_drop(
        inp_tokens,
        inp_lengths, 
        config.inp_drop_prob * drop_decay,
        vocab
    )
    noise_inp_lengths = get_lengths(noise_inp_tokens, eos_idx)

    with torch.no_grad():
        slf_log_probs = model_F(
            noise_inp_tokens, 
            inp_tokens, 
            noise_inp_lengths,
            raw_styles,
            generate=False,
            differentiable_decode=False,
            temperature=temperature,
        )

    slf_rec_loss = loss_fn(slf_log_probs.transpose(1, 2), inp_tokens) * token_mask
    slf_rec_loss = slf_rec_loss.sum() / batch_size
    slf_rec_loss *= config.slf_factor 
    
    with torch.no_grad():
        gen_log_probs = model_F(
            inp_tokens,
            None,
            inp_lengths,
            rev_styles,
            generate=True,
            differentiable_decode=True,
            temperature=temperature,
        )

    gen_soft_tokens = gen_log_probs.exp()
    gen_lengths = get_lengths(gen_soft_tokens.argmax(-1), eos_idx)

    # style consistency loss
    with torch.no_grad():
        adv_log_porbs = model_D(gen_soft_tokens, gen_lengths, rev_styles)
        
    adv_labels = torch.ones_like(rev_styles)
    adv_loss = loss_fn(adv_log_porbs, adv_labels)
    adv_loss = adv_loss.sum() / batch_size
    adv_loss *= config.adv_factor # adv_factor = 1

    # cycle consistency loss
    if not cyc_rec_enable:
        return slf_rec_loss.item(),adv_loss.item()

    with torch.no_grad():
        cyc_log_probs = model_F(
            gen_soft_tokens,
            inp_tokens,
            gen_lengths,
            raw_styles,
            generate=False,
            differentiable_decode=False,
            temperature=temperature,
        )

    cyc_rec_loss = loss_fn(cyc_log_probs.transpose(1, 2), inp_tokens) * token_mask
    cyc_rec_loss = cyc_rec_loss.sum() / batch_size
    cyc_rec_loss *= config.cyc_factor #    

    return slf_rec_loss.item(), cyc_rec_loss.item(), adv_loss.item()

def train(config, vocab, model_F, model_D, train_iters, dev_iters, test_iters,n_global_steps):
    if config.best_model_path:
        print ('Loading model from', config.best_model_path )
        model_F.load_state_dict(torch.load(config.best_model_path + '_F.pth'))#PATH: PATH be behtatin model
        model_D.load_state_dict(torch.load(config.best_model_path+ '_D.pth'))
    else: 
        print('Creating model with fresh parameters.')
    optimizer_F = optim.Adam(model_F.parameters(), lr=config.lr_F, weight_decay=config.L2)
    optimizer_D = optim.Adam(model_D.parameters(), lr=config.lr_D, weight_decay=config.L2)

    his_d_adv_loss = []
    his_f_slf_loss = []
    his_f_cyc_loss = []
    his_f_adv_loss = []    
    global_step = 0
    model_F.train()
    model_D.train()

    config.save_folder = config.save_path 
    os.makedirs(config.save_folder)
    print('Save Path:', config.save_folder)

    print('Training start......')
    def calc_temperature(temperature_config, step):
        num = len(temperature_config) # 1
        #print('len(temperature_config)',num)
        for i in range(num): 
            t_a, s_a = temperature_config[i] #  t_a, s_a =  1, 0
            if i == num - 1:
                return t_a
            t_b, s_b = temperature_config[i + 1]
            if s_a <= step < s_b:
                k = (step - s_a) / (s_b - s_a)
                temperature = (1 - k) * t_a + k * t_b
                return temperature
    batch_iters = iter(train_iters)
    best_loss_dev = float('inf')
    while global_step<n_global_steps: # (kole train_data+/ batchsize) / 5 ke tedad batch ha model_F bara afzayesh shomare global state hast: 833+25=858, ba 25 jame mikonim ke ta akhar batch ha bere
        drop_decay = calc_temperature(config.drop_rate_config, global_step) # drop_decay = 1
        temperature = calc_temperature(config.temperature_config, global_step) # temperature = 1
        batch = next(batch_iters)
        d_adv_loss = d_step(
                config, vocab, model_F, model_D, optimizer_D, batch, temperature
        )
        his_d_adv_loss.append(d_adv_loss)

        # do not back-propagate from the discriminator when it is too poor
        if d_adv_loss < 1.2:
            optimize_total_loss = True
        else:
            optimize_total_loss = False
            
        if config.cyc_rec_enable:
            f_slf_loss, f_cyc_loss, f_adv_loss = f_step(
            config, vocab, model_F, model_D, optimizer_F, batch, temperature, drop_decay, config.cyc_rec_enable, optimize_total_loss
            )
            his_f_cyc_loss.append(f_cyc_loss)
        else:
            f_slf_loss, f_adv_loss = f_step(
            config, vocab, model_F, model_D, optimizer_F, batch, temperature, drop_decay, config.cyc_rec_enable, optimize_total_loss
            )
        his_f_slf_loss.append(f_slf_loss)
        his_f_adv_loss.append(f_adv_loss)

        global_step += 1

        if global_step % config.log_steps == 0: # config.log_steps = 5
            avrg_d_adv_loss = np.mean(his_d_adv_loss)
            avrg_f_slf_loss = np.mean(his_f_slf_loss)
            avrg_f_adv_loss = np.mean(his_f_adv_loss)
            if config.cyc_rec_enable:
                avrg_f_cyc_loss = np.mean(his_f_cyc_loss)            
                log_str = '[iter {}] d_adv_loss: {:.4f}  ' + \
                          'f_slf_loss: {:.4f}  f_cyc_loss: {:.4f}  ' + \
                          'f_adv_loss: {:.4f}  temp: {:.4f}  drop: {:.4f}'
                print(log_str.format(
                    global_step, avrg_d_adv_loss,
                    avrg_f_slf_loss, avrg_f_cyc_loss, avrg_f_adv_loss,
                    temperature, config.inp_drop_prob * drop_decay
                ))
            else:           
                log_str = '[iter {}] d_adv_loss: {:.4f}  ' + \
                          'f_slf_loss: {:.4f}   ' + \
                          'f_adv_loss: {:.4f}  temp: {:.4f}  drop: {:.4f}'
                print(log_str.format(
                    global_step, avrg_d_adv_loss,
                    avrg_f_slf_loss, avrg_f_adv_loss,
                    temperature, config.inp_drop_prob * drop_decay
                ))                
                       
        if global_step % config.eval_steps == 0: #
            his_d_adv_loss = []
            his_f_slf_loss = []
            his_f_cyc_loss = []
            his_f_adv_loss = []
            if config.dev:
                if config.cyc_rec_enable:
                    total_loss, avrg_f_slf_loss, avrg_f_cyc_loss, avrg_f_adv_loss, avrg_d_adv_loss = auto_eval(config, vocab, model_F, model_D, dev_iters, global_step, temperature,drop_decay)
                else:
                    total_loss, avrg_f_slf_loss, avrg_f_adv_loss, avrg_d_adv_loss = auto_eval(config, vocab, model_F, model_D, dev_iters, global_step, temperature,drop_decay)
                if  total_loss < best_loss_dev:
                    best_loss_dev = total_loss
                    print ('saving model...')
                    torch.save(model_F.state_dict(), config.save_folder  + str(global_step) + '_dev_best_model_F.pth')
                    torch.save(model_D.state_dict(), config.save_folder  + str(global_step) + '_dev_best_model_D.pth') 

def test(config, vocab, model_F, model_D, train_iters, dev_iters, test_iters):
    temperature = 1  
    drop_decay = 0
    global_step = 1 
    print ('Loading model from', config.best_model_path )
    model_F.load_state_dict(torch.load(config.best_model_path + '_F.pth'))#PATH: PATH be behtatin model
    model_D.load_state_dict(torch.load(config.best_model_path+ '_D.pth'))
    if config.cyc_rec_enable:
        total_loss, avrg_f_slf_loss, avrg_f_cyc_loss, avrg_f_adv_loss, avrg_d_adv_loss = auto_eval(config, vocab, model_F, model_D, test_iters, global_step, temperature,drop_decay)
        print(total_loss, avrg_f_slf_loss, avrg_f_cyc_loss, avrg_f_adv_loss, avrg_d_adv_loss)
    else:
        total_loss, avrg_f_slf_loss, avrg_f_adv_loss, avrg_d_adv_loss = auto_eval(config, vocab, model_F, model_D, test_iters, global_step, temperature,drop_decay)
        print(total_loss, avrg_f_slf_loss, avrg_f_adv_loss, avrg_d_adv_loss)

def auto_eval(config, vocab, model_F, model_D, test_iters, global_step, temperature,drop_decay):
    model_F.eval()
    model_D.eval()
    # calculating the losses
    his_d_adv_loss = []
    his_f_slf_loss = []
    his_f_cyc_loss = []
    his_f_adv_loss = []
    with torch.no_grad():
        for i, batch in enumerate(test_iters):
            if config.cyc_rec_enable:
                slf_loss, cyc_loss, adv_loss = f_step_test(config, vocab, model_F, model_D, batch, temperature, drop_decay, config.cyc_rec_enable)
                his_f_cyc_loss.append(cyc_loss)
            else: 
                slf_loss,adv_loss = f_step_test(config, vocab, model_F, model_D, batch, temperature, drop_decay, config.cyc_rec_enable)
            d_adv_loss = d_step_test(config, vocab, model_F, model_D, batch, temperature)
            
            his_f_slf_loss.append(slf_loss)
            his_f_adv_loss.append(adv_loss)
            his_d_adv_loss.append(d_adv_loss)
  
        avrg_f_slf_loss = np.mean(his_f_slf_loss) 
        avrg_f_slf_loss *= config.slf_factor # slf_factor = 0.25

        if config.cyc_rec_enable:
            avrg_f_cyc_loss = np.mean(his_f_cyc_loss)
            avrg_f_cyc_loss *= config.cyc_factor # cyc_factor = 0.5

        avrg_f_adv_loss = np.mean(his_f_adv_loss)
        avrg_f_adv_loss *=  config.adv_factor # adv_factor = 1

        avrg_d_adv_loss = np.mean(his_d_adv_loss)

        if config.dev:
            print('*******************************************')
            print('The losses of the whole batches of the dev data')
        elif config.test:
            print('*******************************************')
            print('The losses of the whole batches of the Test data')

        if config.cyc_rec_enable:
            total_loss = avrg_f_adv_loss + avrg_f_cyc_loss + avrg_f_slf_loss
            print('total_loss:{:.4f}, slf_loss:{:.4f}, cyc_loss:{:.4f}, f_adv_loss:{:.4f},  d_adv_loss:{:.4f}'.format( total_loss,avrg_f_slf_loss, avrg_f_cyc_loss, avrg_f_adv_loss,avrg_d_adv_loss))
        else:
            total_loss = avrg_f_adv_loss + avrg_f_slf_loss
            print('total_loss:{:.4f}, slf_loss:{:.4f}, f_adv_loss:{:.4f},  d_adv_loss:{:.4f}'.format( total_loss,avrg_f_slf_loss, avrg_f_adv_loss,avrg_d_adv_loss))
        
    #*****Adding the total loss condition here:**********
    vocab_size = len(vocab)
    eos_idx = vocab.stoi['<eos>']

    def inference(data_iter, raw_style):
        gold_text = []
        raw_output = []
        rev_output = []
        for batch in data_iter:
            inp_tokens = batch.text
            inp_lengths = get_lengths(inp_tokens, eos_idx)
            raw_styles = torch.full_like(inp_tokens[:, 0], raw_style)
            rev_styles = 1 - raw_styles
        
            with torch.no_grad():
                raw_log_probs = model_F(
                    inp_tokens,
                    None,
                    inp_lengths,
                    raw_styles,
                    generate=True,
                    differentiable_decode=False,
                    temperature=temperature,
                )
            
            with torch.no_grad():
                rev_log_probs = model_F(
                    inp_tokens, 
                    None,
                    inp_lengths,
                    rev_styles,
                    generate=True,
                    differentiable_decode=False,
                    temperature=temperature,
                )
                
            gold_text += tensor2text(vocab, inp_tokens.cpu())
            raw_output += tensor2text(vocab, raw_log_probs.argmax(-1).cpu())
            rev_output += tensor2text(vocab, rev_log_probs.argmax(-1).cpu())

        return gold_text, raw_output, rev_output

    pos_iter = test_iters.pos_iter
    neg_iter = test_iters.neg_iter
    gold_text, raw_output, rev_output = zip(inference(neg_iter, 0), inference(pos_iter, 1))

    for k in range(5):
        idx = np.random.randint(len(rev_output[0]))
        print('*' * 20, 'neg sample', '*' * 20)
        print('[gold]', gold_text[0][idx])
        print('[recon ]', raw_output[0][idx])
        print('[trf ]', rev_output[0][idx])

    print('*' * 20, '********', '*' * 20)
    
    for k in range(5):
        idx = np.random.randint(len(rev_output[1]))
        print('*' * 20, 'pos sample', '*' * 20)
        print('[gold]', gold_text[1][idx])
        print('[recon ]', raw_output[1][idx])
        print('[trf ]', rev_output[1][idx])

    print('*' * 20, '********', '*' * 20)

    # save output
    if config.dev:
        save_file_rec = config.save_folder + '/dev_' + str(global_step) + '.rec'
        save_file_tsf = config.save_folder + '/dev_' + str(global_step) + '.tsf'
        eval_log_file = config.save_folder + '/eval_log_dev.txt'
    elif config.test:
        save_file_rec = config.save_path + '/test' + '.rec'
        save_file_tsf = config.save_path + '/test' + '.tsf'
        eval_log_file = config.save_path + '/eval_log_test.txt'

    write_sent( raw_output[0], save_file_rec+ '.0.txt')
    write_sent( raw_output[1], save_file_rec+ '.1.txt')
    write_sent( rev_output[0], save_file_tsf+ '.0.txt')
    write_sent( rev_output[1], save_file_tsf+ '.1.txt')

    model_F.train()
    model_D.train()
    if config.cyc_rec_enable:
        return total_loss, avrg_f_slf_loss, avrg_f_cyc_loss, avrg_f_adv_loss, avrg_d_adv_loss
    else:
        return total_loss, avrg_f_slf_loss, avrg_f_adv_loss, avrg_d_adv_loss
