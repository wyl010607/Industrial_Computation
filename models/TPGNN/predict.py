import torch



def predict_stamp(model, src, stamp, trg):
    '''
    auto regression
    '''
    model.eval()
    with torch.no_grad():
        src_residual = src
        src = src.permute(0, 2, 1, 3)
        trg = trg.permute(0, 2, 1, 3)
        enc_input = model.src_pro(src, stamp)

        # enc_input = model.enc_exp(src)
        # enc_input = model.enc_spa_enco(enc_input)
        # enc_input = model.enc_tem_enco(enc_input)
        stamp_emb = model.stamp_emb(stamp)
        enc_output = model.encoder(enc_input, stamp_emb)

        lines = trg.shape[2]

        trg = torch.zeros(trg.shape).cuda()

        for i in range(lines):
            dec_input = model.trg_pro(trg, enc_output)
            dec_output = model.decoder(dec_input, enc_output)
            dec_output = model.dec_rdu(dec_output)
            trg[:, :, i, :] = dec_output[:, :, i, :]
        return trg.permute(0, 2, 1, 3)
