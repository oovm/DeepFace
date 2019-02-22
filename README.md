📦 Deepface
=======================


### Generate Fake Faces

```py
import DeepFace

f = DeepFace.Generator()
f.bind('MXNet', 'GPU')
f.load().new().save()
```



### Modify Faces

```py
import DeepFace

f = DeepFace.Modifier()
f.bind('PyTorch', 'CPU')
f.load('StarGAN', 'CelebA')
f.infer(img, args)
f.save(path='./imgs/', prefix='mod_')
```



