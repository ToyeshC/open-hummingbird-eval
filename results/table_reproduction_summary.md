# Reproduced results


| Model    | Source                     | 1024×10² (%) | 1024×10³ (%) | 1024×10⁴ (%) | Differences                      |
| -------- | -------------------------- | ------------ | ------------ | ------------ | -------------------------------- |
| ViT-S/16 | Reproduced, batch_size=256 | **37.5**     | **45.0**     | –            | Higher than original             |
|          | Reproduced, batch_size=64  | **37.9**     | **45.1**     | **49.3**     | Up to +2.7% higher than original |
|          | Original                   | 37.2         | 43.1         | 46.6         | –                                |
| ViT-B/16 | Reproduced, batch_size=64  | **48.0**     | **54.7**     | –            | Higher than original             |
|          | Reproduced, batch_size=32  | **47.8**     | **54.6**     | –            | Higher than original             |
|          | Reproduced, batch_size=8   | –            | –            | **57.9**     | Higher than original             |
|          | Original                   | 44.9         | 50.8         | 55.7         | –                                |
| ViT-S/14 | Reproduced, batch_size=64  | **69.6**     | **75.1**     | **77.0**     | Very close to original           |
|          | Original                   | 70.2         | 74.9         | 77.0         | –                                |
| ViT-B/14 | Reproduced, batch_size=64  | **68.0**     | **74.0**     | –            | Slightly lower than original     |
|          | Reproduced, batch_size=8   | –            | –            | **76.6**     | Slightly lower than original     |
|          | Original                   | 69.1         | 74.6         | 76.9         | –                                |
| ViT-L/14 | Reproduced, batch_size=64  | **64.1**     | **71.2**     | –            | Slightly lower than original     |
|          | Reproduced, batch_size=8   | –            | –            | **74.4**     | Slightly lower than original     |
|          | Original                   | 64.6         | 71.7         | 74.8         | –                                |
| ViT-G/14 | Reproduced, batch_size=32  | **62.4**     | –            | –            | Slightly higher than original    |
|          | Reproduced, batch_size=16  | –            | **70.1**     | –            | Slightly higher than original    |
|          | Reproduced, batch_size=8   | –            | –            | **73.3**     | Slightly lower than original     |
|          | Original                   | 62.3         | 69.9         | 73.6         | –                                |
