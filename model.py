# model.py
import torch
import torch.nn as nn

class AudioVisualFusionModel(nn.Module):
    def __init__(self, audio_dim=40, video_dim=2054, d_model=256, nhead=4, num_layers=2, num_classes=2, dropout=0.3):
        super().__init__()
        self.audio_proj = nn.Linear(audio_dim, d_model)
        self.video_proj = nn.Linear(video_dim, d_model)
        self.fuse_proj = nn.Linear(2 * d_model, d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True, dropout=dropout
        )
        self.fusion_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.cross_attn_audio_to_video = nn.MultiheadAttention(d_model, nhead, batch_first=True, dropout=dropout)
        self.cross_attn_video_to_audio = nn.MultiheadAttention(d_model, nhead, batch_first=True, dropout=dropout)

        fusion_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True, dropout=dropout
        )
        self.fusion_transformer = nn.TransformerEncoder(fusion_layer, num_layers=2)

        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )

    def forward(self, audio, video):
        audio = self.audio_proj(audio)
        video = self.video_proj(video)
        fused = torch.cat([audio, video], dim=-1)
        fused = self.fuse_proj(fused)
        B = fused.size(0)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        fused_seq = torch.cat([cls_tokens, fused], dim=1)
        enc = self.fusion_encoder(fused_seq)
        enc_audio = enc[:, 1:, :]
        enc_video = enc[:, 1:, :]
        a2v, _ = self.cross_attn_audio_to_video(enc_audio, enc_video, enc_video)
        v2a, _ = self.cross_attn_video_to_audio(enc_video, enc_audio, enc_audio)
        fused_seq = torch.cat([cls_tokens, a2v + v2a], dim=1)
        fused_out = self.fusion_transformer(fused_seq)
        cls_out = fused_out[:, 0, :]
        logits = self.classifier(cls_out)
        return logits
