📦 Deepface
=======================


### Generate Fake Faces

```py
import DeepFace

g = DeepFace.Generator()
g.bind('MXNet', 'GPU')
g.load().new().save()
```



### Modify Faces

```py
import DeepFace

g = DeepFace.Modifier()
g.bind('PyTorch', 'CPU')
g.load('StarGAN', 'CelebA')
g.infer(img, args)
g.save(path='./imgs/', prefix='mod_')
```



