# Introduction
>In this paper we propose a generative probabilistic model for stereo matching, called ELAS
This is a stereo matching solution that reduces ambiguities by building a prior over disparity space. We consider a set of points called support points that are likely to be more reliable to match with the other images pixels because they are chosen from robust matching techniques
Forming a triangulation, where support points are the vertices and the triangles are faces.
Within each triangular region, disparity varies linearly

# Terminologies
- Binocular Stereo Matching: Matching two pixels from different 2d images.
- aggregation window: a square window traditionally used to compare patches between left and right images
