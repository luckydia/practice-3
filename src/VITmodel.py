import torch
import torch.nn.functional as F
from torch import nn
import lightning as L
from torchvision.transforms import Compose, Resize, ToTensor
from einops import rearrange, reduce, repeat, einsum
from einops.layers.torch import Rearrange, Reduce
from torchmetrics.functional import accuracy


class PatchEmbedding(L.LightningModule):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=64, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()

        self.positions = nn.Parameter(torch.randn(1, (img_size // patch_size) ** 2 + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        self.projection = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim, kernel_size = patch_size, stride = patch_size),
            Rearrange('b e h w -> b (h w) e')
        )

    def forward(self, image):
        b, c, h, w = image.shape
        image = self.projection(image)
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b = b)

        image = torch.cat([cls_tokens, image], dim=1)
        image += self.positions
        return image


class MLP(L.LightningModule):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0., act_layer = nn.GELU):
        super().__init__()

        # Linear Layers
        self.model = nn.Sequential(
            nn.Linear(768, 3072),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(3072, 768),
            nn.Dropout(drop)
        )


    def forward(self, x):
        x = self.model(x)
        return x


class Attention(L.LightningModule):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., out_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(attn_drop)
        self.out = nn.Linear(dim, dim)
        self.out_drop = nn.Dropout(out_drop)
        self.act = nn.Softmax(dim=-1)

    def forward(self, x):
        b, n, *_ = x.shape
        h = self.num_heads
        # Attention
        qkv = self.qkv(x)
        q, k, v = rearrange(qkv, 'b n (qkv h d) -> qkv b h n d', qkv=3, h=h)
        attn = einsum(q, k, 'a b c d, a b f d -> a b c f') * self.scale
        attn = self.act(attn)
        attn = self.attn_drop(attn)
        # Out projection
        out = einsum(attn, v, 'a b c d, a b d f -> a b c f')
        out = rearrange(out, 'a b c d -> a c (b d)')
        out = self.out_drop(self.out(out))
        return out


class Block(L.LightningModule):
    def __init__(self, dim, num_heads=8, mlp_ratio=4, drop_rate=0.):
        super().__init__()

        # Normalization
        self.norm1 = nn.LayerNorm(dim)

        # Attention
        self.attn = Attention(dim, num_heads=num_heads)

        # Dropout
        self.drop = nn.Dropout(drop_rate)

        # Normalization
        self.norm2 = nn.LayerNorm(dim)

        # MLP
        self.mlp = MLP(dim, int(dim * mlp_ratio))


    def forward(self, x):
        # Attetnion
        x = self.norm1(x)
        x = self.drop(self.attn(x))
        x = self.norm2(x)

        # MLP
        x = self.mlp(x)
        return x


class Transformer(L.LightningModule):
    def __init__(self, depth, dim, num_heads=8, mlp_ratio=4, drop_rate=0.):
        super().__init__()
        self.blocks = nn.ModuleList([
            Block(dim, num_heads, mlp_ratio, drop_rate)
            for i in range(depth)])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class ViT(L.LightningModule):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self, img_size=64, patch_size=16, in_chans=3, num_classes=10,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.,
                 qkv_bias=False, drop_rate=0., ):
        super().__init__()

        # Присвоение переменных
        self.num_classes = num_classes
        self.learning_rate = 0.0001
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.drop_rate = drop_rate

        # Path Embeddings, CLS Token, Position Encoding
        self.patch_embed = PatchEmbedding(img_size=self.img_size,
                                          patch_size=self.patch_size,
                                          in_chans=self.in_chans,
                                          embed_dim=self.embed_dim)

        # Transformer Encoder
        self.transformer = Transformer(self.depth, self.embed_dim, self.num_heads, self.mlp_ratio, self.drop_rate)

        # Classifier
        self.classifier = nn.Linear(self.embed_dim,
                                    self.num_classes)  # MLP(embed_dim, int(embed_dim * mlp_ratio), self.num_classes)

    def forward(self, x):
        # Path Embeddings, CLS Token, Position Encoding
        x = self.patch_embed(x)
        # Transformer Encoder
        x = self.transformer(x)
        # Classifier
        x = self.classifier(x[:, 0, :])

        return x

    def training_step(self, batch):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        print(x.shape, y.shape)
        logits = self(x)
        print(logits.shape, y.shape)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y, task="multiclass", num_classes=10)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer