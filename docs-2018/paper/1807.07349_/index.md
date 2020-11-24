# https://arxiv.org/abs/1807.07349v1
Generative Adversarial Networks for MR-CT Deformable Image Registration

用于MR-CT可变形图像配准的生成对抗网络

## Abstract
Deformable Image Registration (DIR) of MR and CT images is one of the most challenging registration task, due to the inherent structural differences of the modalities and the missing dense ground truth. Recently cycle Generative Adversarial Networks (cycle-GANs) have been used to learn the intensity relationship between these 2 modalities for unpaired brain data. Yet its usefulness for DIR was not assessed. In this study we evaluate the DIR performance for thoracic and abdominal organs after synthesis by cycle-GAN. We show that geometric changes, which differentiate the two populations (e.g. inhale vs. exhale), are readily synthesized as well. This causes substantial problems for any application which relies on spatial correspondences being preserved between the real and the synthesized image (e.g. plan, segmentation, landmark propagation). To alleviate this problem, we investigated reducing the spatial information provided to the discriminator by decreasing the size of its receptive fields. Image synthesis was learned from 17 unpaired subjects per modality. Registration performance was evaluated with respect to manual segmentations of 11 structures for 3 subjects from the VISERAL challenge. State-of-the-art DIR methods based on Normalized Mutual Information (NMI), Modality Independent Neighborhood Descriptor (MIND) and their novel combination achieved a mean segmentation overlap ratio of 76.7, 67.7, 76.9%, respectively. This dropped to 69.1% or less when registering images synthesized by cycle-GAN based on local correlation, due to the poor performance on the thoracic region, where large lung volume changes were synthesized. Performance for the abdominal region was similar to that of CT-MRI NMI registration (77.4 vs. 78.8%) when using 3D synthesizing MRIs (12 slices) and medium sized receptive fields for the discriminator.

MR和CT图像的可变形图像配准（DIR）是最具挑战性的配准任务之一，因为模态的固有结构差异和缺失的密集地面实况。最近循环生成性对抗网络（cycle-GANs）已被用于学习这两种不成对脑数据模态之间的强度关系。然而，没有评估其对DIR的有用性。在本研究中，我们评估了通过cycle-GAN合成后胸部和腹部器官的DIR性能。我们表明，区分两个种群（例如吸气与呼气）的几何变化也很容易合成。这对于依赖于在真实和合成图像之间保留空间对应的任何应用（例如，平面图，分割，界标传播）而言都会引起实质性问题。为了缓解这个问题，我们研究了通过减小​​其感知域的大小来减少提供给鉴别器的空间信息。每种方式从17个未配对的受试者中学习图像合成。针对来自VISERAL挑战的3个受试者的11个结构的手动分割评估了登记性能。基于归一化互信息（NMI），模态独立邻域描述符（MIND）及其新颖组合的现有技术DIR方法实现了平均分段重叠率分别为76.7,67.7,76.9％。当基于局部相关性记录由cycle-GAN合成的图像时，由于在合成大的肺容积变化的胸部区域上的不良性能，这下降到69.1％或更低。当使用3D合成MRI（12个切片）和中等大小的感受野用于鉴别器时，腹部区域的性能类似于CT-MRI NMI配准（77.4对78.8％）。

## Introduction
Deformable Image Registration (DIR) is a challenging task and active field of research in medical image analysis [1]. Its main application is fusion of the information acquired by the different modalities to facilitate diagnosis and treatment planning [1]. For example, in radiotherapy treatment planning Magnet Resonance (MR) images are used to segment the tumor and organs at risk, while the tissue density information provided by the corresponding Computer Tomography (CT) image is used for dose planning [2]. CT and MR images are acquired using separate devices and often on different days. Therefore the patient will not be in exactly the same posture and the position of inner organs might change, due to respiration, peristalsis, bladder filling, gravity, etc. Thus, DIR is needed. The main difficulty of MR-CT DIR is the definition of an image similarity measure, which reliably quantifies the local image alignment for optimizing the many free parameters of the spatial transformation. This is an inherent problem as multimodal images are acquired because they provide complementary information.

可变形图像配准（DIR）是医学图像分析中具有挑战性的任务和活跃的研究领域[1]。其主要应用是融合不同方式获得的信息，以促进诊断和治疗计划[1]。例如，在放射治疗计划中，磁共振（MR）图像用于分割处于危险中的肿瘤和器官，而由相应的计算机断层扫描（CT）图像提供的组织密度信息用于剂量计划[2]。使用单独的设备并且通常在不同的日子获取CT和MR图像。因此，由于呼吸，蠕动，膀胱充盈，重力等，患者将不会处于完全相同的姿势并且内部器官的位置可能改变。因此，需要DIR。MR-CT DIR的主要难点在于图像相似性度量的定义，其可靠地量化局部图像对准以优化空间变换的许多自由参数。这是一个固有的问题，因为它们提供了补充信息，因此获得了多模态图像。

Multi-modal similarity measures. The main voxel-wise multi-modal image (dis)similarity measures are (i) statistical measures that use intensity information directly and try to maximize (non-linear) statistical dependencies between the intensities of the images (e.g. Normalized Mutual Information (NMI) [3], MI [4]), and (ii) structural measures based on structural representations that try to be invariant to different modalities (e.g. normalized gradient fields [5], entropy images [6], Modality Independent Neighborhood Descriptor (MIND) [7]).

多模态相似性度量。主体素多模态图像（dis）相似性度量是（i）直接使用强度信息并尝试最大化（非线性）图像强度之间的统计依赖性的统计量度（例如，归一化互信息（NMI） [3]，MI [4]）和（ii）基于结构表示的结构测量，试图对不同的模态不变（例如归一化梯度场[5]，熵图像[6]，模态独立邻域描述符（MIND）[7]）。

Intensity remapping. The drawback of structural representations is that all unstructured (e.g. homogenous) regions are mapped to the same representation regardless of their original intensity. To avoid this information reduction, methods to directly re-map intensities have been proposed [8,6,9]. The joint histogram of the coarsely registered images was employed to remap the intensities of both images to a common modality based on the least conditional variance to remove structures not visible in both images [8]. Assuming that the global self-similarity of the images (i.e. the similarities between all image patches) is preserved across modalities, intensities were mapped into a 1D Laplacien Eigenmap based on patch intensity similarity [6]. A k-means clustering based binning scheme, to remap spatially unconnected components with similar intensities to distinct intensities, was proposed in [9] for retina images. While these intensity-remappings provide some improvements, they are simplifications to the underlying complex relationship between the intensity of the two modalities.

强度重映射。结构表示的缺点是所有非结构化（例如同质）区域都被映射到相同的表示，而不管它们的原始强度如何。为了避免这种信息减少，已经提出了直接重新绘制强度的方法[8,6,9]。使用粗略配准图像的联合直方图将两个图像的强度重新映射到基于最小条件方差的共同模态，以去除两个图像中不可见的结构[8]。假设图像的全局自相似性（即所有图像块之间的相似性）在模态之间被保留，则基于斑块强度相似性将强度映射到1D拉普拉斯算子本征图[6]。在[9]中针对视网膜图像提出了基于k均值聚类的分箱方案，以将具有相似强度的空间不连接分量重新映射到不同强度。虽然这些强度重新映射提供了一些改进，但它们简化了两种模态强度之间潜在的复杂关系。

Learning from paired data. Given aligned multi-modal training data, attempts have been made to learn this complex relationship. The last layer of a deep neural network (DNN) classifier, which discriminates between matching and not matching patches, was used to directly learn the similarity measure [10]. The DNN was initialized by a stacked denoised autoencoder, where the lower layers were separately trained per modality to get modality-dependent features. It was observed that the learned CT filters look mostly like edge-detectors, while the MR filters detect more complex texture features. In [11] the expected joint intensity distribution was learned from co-registered images. The dissimilarity measure was then based on the Bhattacharyya distance between the expected and observed distribution. Machine learning has been used to learn how to map one modality to the other. [12] synthesized CT from MR brain images by matching MR patches to an atlas (created from co-registered MR and CT images) and augmented these by considering all convex patch combinations. [13] proposed a bi-directional image synthesis approach for non-rigid registration of the pelvic area, where random forests are trained on Haar-like features extracted from pairs of pre-aligned CT and MR patches. An auto-context model was used to incorporate neighboring prediction results. All these learning-based approaches depend on co-registered multi-modal images for training. This is very difficult for deforming structures as dense (voxel-wise) spatial correspondences are required and CT and MR images cannot be acquired simultaneously yet [14].

从配对数据中学习。给定对齐的多模态训练数据，已经尝试学习这种复杂的关系。深度神经网络（DNN）分类器的最后一层，用于区分匹配和不匹配的补丁，用于直接学习相似性度量[10]。DNN由堆叠的去噪自动编码器初始化，其中较低层按模态分别训练以获得模态相关的特征。据观察，学习的CT滤波器看起来大部分像边缘检测器，而MR滤波器检测更复杂的纹理特征。在[11]中，预期的关节强度分布是从共同登记的图像中学习的。然后，相异性度量基于预期分布和观察分布之间的Bhattacharyya距离。机器学习已被用于学习如何将一种模态映射到另一种模态。[12]通过将MR贴片与图谱（由共同登记的MR和CT图像创建）匹配，从MR脑图像合成CT，并通过考虑所有凸形斑块组合来增强这些。[13]提出了一种双向图像合成方法，用于骨盆区域的非刚性配准，其中随机森林训练从从预对齐的CT和MR贴片对中提取的Haar样特征。使用自动上下文模型来合并相邻预测结果。所有这些基于学习的方法都依赖于共同注册的多模态图像进行训练。这对于变形结构是非常困难的，因为需要密集（体素方向）空间对应并且CT和MR图像不能同时获取[14]。

Learning without paired data. A cross-modality synthesis method which does not require paired data was proposed in [15]. It is based on generating multiple target modality candidate values for each source voxel independently using cross-modal nearest neighbor search. A global solution is then found by simultaneously maximizing global MI and local spatial consistency. Finally, a coupled sparse representation was used to further refine the synthesized images. When applied to T1/T2 brain MRIs, T1 images were better synthesized than T2 images (0.93 vs. 0.85 correlation to ground truth). Extending the method to a supervised setting outperformed state-of-the-art supervised methods slightly.

学习没有配对数据。在[15]中提出了一种不需要配对数据的跨模态合成方法。它基于使用跨模态最近邻搜索独立地为每个源体素生成多个目标模态候选值。然后通过同时最大化全局MI和局部空间一致性找到全局解决方案。最后，使用耦合稀疏表示来进一步细化合成图像。当应用于T1/T2脑MRI时，T1图像比T2图像更好地合成（0.93对0.85与地面事实的相关性）。将方法扩展到监督设置略微优于现有技术的监督方法。

Recently cyclic-consistent Generative Adversarial Networks (cycle-GANs) were proposed for learning an image-to-image mapping between two domains (A&B) from unpaired datasets [16]. The method is based on two generator networks (GB to synthesize image ˆIB from IA, GA) and two discriminator networks (DA, DB). Besides the usual discriminator loss to differentiate synthesized and real images (e.g. ˆIA, IA), a cycle loss was introduced which measures the difference between the real image and its twice synthesized image, e.g. |IA − GA(GB(IA))|1. Good performances were shown for various domain translation tasks like labels to photos and arial photos to maps. Very recently, this promising approach was employed for slice-wise synthesizing CT from MR head images from unpaired data [2]. It achieved lower mean squared errors (74 vs. 89 HU) than when training the same generator network on rigidly aligned MR and CT data [17]. It was reasoned that this could be due to misalignments, as the images contained also deforming structures (e.g. neck, mouth). CycleGANs were used for synthesis of MR from unpaired CT images for enriching a cardiac dataset for training thereafter a segmentation network [18]. A view alignment step using the segmentations was incorporated to make the layout (e.g. position, size of anatomy) of the CT and MR images similar, such that the discriminator cannot use the layout to differentiate between them. Furthermore the myocardium mask for both modalities was provided during training, as the cycle-GAN not only changed the intensities but also anatomical locations such that the mask was no longer in correspondence with the image. Hence this is not a completely unsupervised approach. Similarly, a shape-consistency loss from segmentations was incorporated in [19] to avoid geometric changes between the real and synthesized images. It was argued that ”from the discriminator perspective, geometric transformations do not change the realness of synthesized images since the shape of training data is arbitrary”. However this does not hold if there is a geometric bias between the two datasets.

最近提出了循环一致的生成对抗网络（cycle-GAN）用于学习来自未配对数据集的两个域（A和B）之间的图像到图像映射[16]。该方法基于两个发生器网络（GB以合成来自IA，GA的图像IB）和两个鉴别器网络（DA，DB）。除了用于区分合成和真实图像（例如IA，IA）的通常的鉴别器损失之外，还引入了循环损失，其测量实际图像与其两次合成图像之间的差异，例如，`|IA - GA（GB（IA））|`。各种领域翻译任务的良好表现，如照片标签和地图照片。最近，这种有前景的方法被用于从不成对数据的MR头部图像中切片合成CT [2]。与在刚性对齐的MR和CT数据上训练相同的发电机网络相比，它实现了更低的均方误差（74对89 HU）[17]。据推测，这可能是由于未对准造成的，因为图像中还包含变形结构（例如颈部，嘴部）。CycleGAN用于从未配对的CT图像合成MR，用于丰富心脏数据集，用于此后训练分割网络[18]。结合使用分割的视图对准步骤以使CT和MR图像的布局（例如，解剖结构的位置，尺寸）相似，使得鉴别器不能使用布局来区分它们。此外，在训练期间提供两种模态的心肌掩模，因为cycle-GAN不仅改变了强度而且改变了解剖位置，使得掩模不再与图像对应。因此，这不是一种完全无监督的方法。类似地，在[19]中结合了来自分割的形状一致性损失，以避免真实和合成图像之间的几何变化。有人认为，“从鉴别器的角度来看，几何变换不会改变合成图像的真实性，因为训练数据的形状是任意的”。但是，如果两个数据集之间存在几何偏差，则不成立。

Synthesized MR PD/T1 brain images via patch matching were shown to be useful for segmentation and inter-modality cross-subject registration [20]. If this also holds for MR-CT synthesis via cycle-GANs for thoracic and abdominal regions has not yet been studied. Our contributions include (i) combining two state-of-the-art multi-modal DIR similarity measures (NMI, MIND), (ii) studying the effect of the image region size on the consistency of the synthesized 3D images, and (iii) evaluating the usefulness of synthesized images for deformable registration of CT and MR images from the thorax and abdomen against a strong baseline.

通过斑块匹配合成的MR PD / T1脑图像显示可用于分割和跨模式跨主题登记[20]。如果这也适用于通过循环的MR-CT合成 - 胸部和腹部区域的GAN尚未研究。我们的贡献包括（i）结合两种最先进的多模态DIR相似性度量（NMI，MIND），（ii）研究图像区域大小对合成3D图像一致性的影响，以及（iii）评估合成图像对于胸部和腹部的CT和MR图像的可变形配准对强基线的有用性。

## Materials
We used 17 unpaired and 3 paired 3D MR-CT images from the VISCERAL Anatomy3 benchmark training set (unenhanced, whole body, MR-T1) and their gold standard segmentations for evaluation [21]. The 3 subjects with paired data had IDs 10000021, 10000067 and 10000080. All MRIs were bias field corrected using the N4ITK method [22]. All images were resampled to an isotropic resolution of 1.25 mm. This was motivated by the image resolution of the original MRIs being 1.25×6×1.25 mm3 in left-right, posterior-anterior and superiorinferior direction. The CT images had a resolution between 0.8×0.8×1.5 mm3 and 1.0×1.0×1.5 mm3 .

我们使用来自VISCERAL Anatomy3基准训练集（未增强，全身，MR-T1）的17个未配对和3个配对的3D MR-CT图像及其用于评估的金标准分割[21]。配对数据的3名受试者的ID为10000021,10000067和10000080.所有MRI均使用N4ITK方法进行偏场校正[22]。所有图像都重新采样为1.25 mm的各向同性分辨率。这是由于原始MRI的图像分辨率在左右，后 - 前和下下方向为1.25×6×1.25mm3。CT图像的分辨率在0.8×0.8×1.5mm 3和1.0×1.0×1.5mm 3之间。

To reduce memory requirements, we automatically extracted from each image two regions such that the included gold standard segmentations were at least 5 mm away from the inferior and superior region boundary. The thorax region covered the segmentations of the liver, spleen, gallbladder, and right and left lung. The abdominal region contained the bladder, lumbar vertebra 1, right and left kidney, and right and left psoas major muscle, see Figs. 2, 1, left column.

为了减少存储器需求，我们自动从每个图像中提取两个区域，使得所包括的金标准分割距离下部和上部区域边界至少5mm。胸部区域覆盖肝，脾，胆囊和左右肺的分割。腹部区域包含膀胱，腰椎1，右肾和左肾，以及左右腰大肌，见图。2,1，左栏。

Closer investigation of poor performing registration results showed that for case 10000067 the segmentation labels of the right and left kidney were swapped in the MRI. Additionally, for 10000080 the segmentation of the lumbar vertebra 1 in the MRI seems to be that of lumbar vertebra 2. We corrected these kidney annotations and excluded this lumbar vertebra 1 segmentations from the results.

对表现不佳的登记结果的仔细研究表明，对于病例10000067，右肾和左肾的分割标签在MRI中交换。另外，对于10000080，MRI中的腰椎1的分割似乎是腰椎2的分割。我们校正了这些肾脏注释并从结果中排除了这种腰椎1的分割。

## Method

### Image Synthesis
Cycle-GAN. For image synthesis, we followed the cycle-GAN network architecture as described in [16,2], starting from an existing implementation1 . In short, the two generators (GCT, GMR) are 2D fully convolutional networks with 9 residual blocks and two fractionally strided convolution layers (res-net). The discriminators (DCT, DMR) are fully convolutional architectures to classify overlapping P×P image patches as real or fake (PatchGAN) [23]2 .

Cycle-GAN。对于图像合成，我们遵循[16,2]中描述的cycle-GAN网络架构，从现有的实现开始。简而言之，两个发生器（GCT，GMR）是2D完全卷积网络，具有9个残余块和两个分数跨度卷积层（res-net）。鉴别器（DCT，DMR）是完全卷积体系结构，用于将重叠的P×P图像块分类为真实或伪造（PatchGAN）[23]。

The networks take input images of size 256×256 pixels and C channels. Larger-sized test images were synthesized from the average result of 256×256×C regions extracted with a stride length of S×S×SC . The cycle-GAN was optimized to reduce the overall loss L, which is a weighted sum of the discriminator losses LCT, LMR and the generator cyclic loss Lcyc:

网络采用大小为256×256像素和C通道的输入图像。从以步长S×S×SC提取的256×256×C区域的平均结果合成较大尺寸的测试图像。cycle-GAN经过优化以降低总损耗L，它是鉴别器损耗LCT，LMR和发电机循环损耗Lcyc的加权和：

$$
\begin{aligned}
L = L_{CT} + L_{MR} + \lambda_{cyc} L_{cyc} \\
L_{CT} = (1 - D_{CT}(I_{CT}))^2 + D_{CT}(G_{CT}(I_{MR}))^2 \\
L_{MR} = (1 - D_{MR}(I_{MR}))^2 + D_{MR}(G_{MR}(I_{CT}))^2 \\
L_{cyc} = \| G_{CT}(G_{MR}(I_{CT})) - I_{CT} \| + \| G_{MR}(G_{CT}(I_{MR})) -I_{MR} \|
\end{aligned}
$$

### Image Registration
Rigid Registration. The CT and MR images were first rigidly registered using the function imregister from the MATLAB Image Processing Toolbox [24], set to multi-modal configuration (Mattes Mutual Information, one-plus-one evolutionary optimization). These rigid registration results were then used as starting points for all subsequent deformable image registrations.

刚性注册。CT和MR图像首先使用MATLAB图像处理工具箱[24]中的函数imregister进行严格注册，设置为多模态配置（Mattes Mutual Information，一加一进化优化）。然后将这些刚性配准结果用作所有后续可变形图像配准的起始点。

Deformable Registration - MIND. The so-called modality independent neighborhood descriptor (MIND) was proposed as dissimilarity measure for multi-modal DIR [7]. MIND is based on a multi-dimensional descriptor sMIND per voxel x, which captures the self-similarity of the image patch around x (denoted as P(x)) with the patches P(x + r) in a local neighborhood N of x. The single entries sMIND(I, x, ri) are calculated by a Gaussian function

可变形注册 - MIND。提出了所谓的模态独立邻域描述符（MIND）作为多模态DIR的相异度度量[7]。MIND基于每个体素x的多维描述符sMIND，其捕获x周围的图像块（表示为P（x））与x的局部邻域N中的块P（x+r）的自相似性。单个条目sMIND（I，x，ri）由高斯函数计算

$$
\begin{aligned}
S_{MIND}(I,x,r_i) = \frac{1}{n} exp(- \frac{d_p(I,x,r_i)}{v(I,x)})
\end{aligned}
$$

where n is a normalization constant such that the maximum value in sMIND is 1, dp defines the patch dissimilarity dp(I, x, r)=SUM xj∈P(x) Gσ(xj)(P(xj) − P(xj+r))^2 with Gaussian kernel Gσ of the same size as patch P(x) and the halfsize of the patch being equal to [1.5σ]. v is the variance of a six-neighborhood search region. sMIND is calculated in a dense fashion for each image independently. The dissimilarity EMIND(A, B) of images A and B is finally defined by EMIND(A, B)=SUM x∈Ω EMIND(A, B, x)^2 with

$$
\begin{aligned}
E_{MIND}(A,B,x) = \frac{1}{|\mathcal{N}|} \sum_{r_i \in \mathcal{N}} | S_{MIND}(A,x,r_i) - S_{MIND}(B,x,r_i) |
\end{aligned}
$$

其中n是归一化常数，使得sMIND中的最大值为1，dp定义补丁相异度dp（I，x，r）=SUM xj∈P（x）Gσ（xj）（P（xj）-P（xj）+r））^2，其高斯核Gσ与补丁P（x）大小相同，并且补丁的半尺寸等于[1.5σ]。v是六邻域搜索区域的方差。sMIND以密集的方式独立计算每个图像。图像A和B的相异性EMIND（A，B）最终由EMIND（A，B）=SUMx∈ΩEMIND（A，B，x）^2定义。

In the MIND registration framework, the images are downsampled via Gaussian pyramids and the deformation field is regularized via the squared L2-norm. Additionally after each Gauss-Newton update step during optimization, each deformation field is replaced by combining half of its own transformation with half of the inverse transformation of the other deformation field (see [7,25]) to obtain diffeomorphic transformations. We used the provided code [26] and compared results after integrating the MIND measure in our DIR method [27].

在MIND注册框架中，通过高斯金字塔对图像进行下采样，并且通过平方L2范数对变形场进行正则化。另外，在优化期间的每个Gauss-Newton更新步骤之后，通过将其自身变换的一半与另一变形场的逆变换的一半（参见[7,25]）组合来替换每个变形场以获得微分变形。我们使用提供的代码[26]并在我们的DIR方法[27]中整合MIND测量后比较结果。

Deformable Registration - ourDIR. We extended our DIR method, based a linearly interpolated grid of control points and various displacement regularization measures, to incorporate the multi-modal (dis)similarity measures normalized mutual information (NMI) and MIND, and their combination NMI+MIND.

可变形注册 - 我们DIR。我们扩展了我们的DIR方法，基于线性插值控制点网格和各种位移正则化测量，以结合多模态（dis）相似性度量标准化互信息（NMI）和MIND，以及它们的组合NMI + MIND。

Given fixed image I_f, moving image I_m, displacements k at the control points, and interpolation function d to get dense displacements, the NMI dissimilarity ENMI(I_f, I_m(d(k))) is defined by ENMI(A, B)=−(HA + HB)/HAB, with marginal entropies HA, HB and joint entropy HAB computed from intensity histograms with 100 equally-spaced bins between the 0.5 and 99.5 percentiles of the image intensity. The gradients of ENMI(I_f, I_m(d(k))) with respect to d(k)^i[x] are calculated as described in [28]. To avoid infinity gradients, we replace zero probabilities with 1/(2N_V), where N_V is the number of image voxels. We combined the dissimilarities NMI and MIND by

$$
\begin{aligned}
E_{N+M}(I_f, I_m(d(k))) = \beta E_{NMI}(I_f, I_m(d(k))) + (1 - \beta) s E_{MIND}(I_f, I_m(d(k)))
\end{aligned}
$$

给定固定图像I_f，运动图像I_m，控制点处的位移k和插值函数d以获得密集位移，NMI相异度ENMI（I_f，I_m（d（k）））由ENMI（A，B）=-（HA+HB）/HAB，具有边缘熵HA，HB和联合熵HAB，根据强度直方图计算，在图像强度的0.5和99.5百分位之间具有100个等距离的区间。如[28]中所述计算ENMI（I_f，I_m（d（k）））相对于d（k）^i[x]的梯度。为了避免无限渐变，我们用1/（2N_V）替换零概率，其中N_V是图像体素的数量。我们将不同的NMI和MIND结合起来

where s is a scaling parameter to get EMIND in the same range [29] and β∈[0, 1] is a weighting term. The choice of s is not trivial, as the magnitude of change per dissimilarity measure from initial to ideal knot displacements D_{init,ideal}(E_{dissim}) is unknown. We tested 3 strategies, namely (i) using a fixed parameter s, (ii) using the initial gradient magnitude via, (iii) basing it on the change in dissimilarity during registration

其中s是一个缩放参数，用于获得相同范围内的EMIND[29]，而β∈[0,1]是一个加权项。s的选择并非无足轻重，因为每个不相似度的变化幅度从初始到理想的结位移D_{init，ideal}（E_{dissim}）是未知的。我们测试了3种策略，即（i）使用固定参数s，（ii）使用初始梯度幅度，（iii）基于注册期间相异性的变化

最终损失函数是:

$$
\begin{aligned}
F(d(k)) = E_{dissim}(I_f, I_m(d(k))) + \lambda R(k)
\end{aligned}
$$

其中`R(k)`通过TV或L2范数使控制点处的位移正规化。

## Experiments and Results
