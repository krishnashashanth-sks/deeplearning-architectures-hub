import torch.nn as nn
from layers import Encoder,Decoder,Postnet

class Tacotron2(nn.Module):
  def __init__(self,hparams):
    super(Tacotron2,self).__init__()
    self.n_mel_channels=hparams.n_mel_channels
    self.n_frames_per_step=hparams.n_frames_per_step
    self.encoder=Encoder(hparams)
    self.decoder=Decoder(hparams)
    self.postnet=Postnet(hparams)
  def forward(self,text_inputs,text_lengths,mel_targets,mel_lengths):
    text_lengths=text_lengths.long()
    mel_lengths=mel_lengths.long()
    encoder_outputs=self.encoder(text_inputs)
    mel_targets_reshaped=mel_targets.view(
        mel_targets.size(0),self.n_mel_channels*self.n_frames_per_step,-1
    ).transpose(1,2)
    mel_outputs,gate_outputs,alignments=self.decoder(
        encoder_outputs,mel_targets_reshaped.transpose(0,1),text_lengths # Added .transpose(0,1) here
    )
    mel_outputs_postnet=self.postnet(mel_outputs)
    mel_outputs_postnet=mel_outputs+mel_outputs_postnet
    return mel_outputs,mel_outputs_postnet,gate_outputs,alignments
  def inference(self,text_inputs,text_lengths):
    text_lengths=text_lengths.long()
    encoder_outputs=self.encoder(text_inputs)
    mel_outputs,gate_outputs,alignments=self.decoder.inference(
        encoder_outputs,text_lengths) # Corrected to call decoder.inference
    mel_outputs_postnet=self.postnet(mel_outputs)
    mel_outputs_postnet=mel_outputs+mel_outputs_postnet # Fixed: Added residual connection
    return mel_outputs,mel_outputs_postnet,gate_outputs,alignments