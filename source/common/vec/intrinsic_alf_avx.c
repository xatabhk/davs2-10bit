/*****************************************************************************
*  Copyright (C) 2016 uavs2dec project,
*  National Engineering Laboratory for Video Technology(Shenzhen),
*  Digital Media R&D Center at Peking University Shenzhen Graduate School, China
*  Project Leader: Ronggang Wang <rgwang@pkusz.edu.cn>
*
*  Main Authors: Zhenyu Wang <wangzhenyu@pkusz.edu.cn>, Kui Fan <kuifan@pku.edu.cn>
*               Shenghao Zhang <1219759986@qq.com>， Bingjie Han, Kaili Yao, Hongbin Cao,  Yueming Wang,
*               Jing Su, Jiaying Yan, Junru Li
*
* This program is free software; you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation; either version 2 of the License, or
* (at your option) any later version.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with this program; if not, write to the Free Software
* Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02111, USA.
*
* This program is also available under a commercial proprietary license.
* For more information, contact us at rgwang@pkusz.edu.cn.
*****************************************************************************/

#include <mmintrin.h>
#include <emmintrin.h>
#include <tmmintrin.h>
#include <smmintrin.h>
#include <immintrin.h>

#include "../common.h"
#include "intrinsic.h"


ALIGN8(int32_t alf_mask256[7][8]) = {
        { -1, 0, 0, 0, 0, 0, 0, 0 },
        { -1, -1, 0, 0, 0, 0, 0, 0 },
        { -1, -1, -1, 0, 0, 0, 0, 0 },
        { -1, -1, -1, -1, 0, 0, 0, 0 },
        { -1, -1, -1, -1, -1, 0, 0, 0 },
        { -1, -1, -1, -1, -1, -1, 0, 0 },
        { -1, -1, -1, -1, -1, -1, -1, 0 }
};

#if HIGH_BIT_DEPTH

void alf_filter_block_avx2_10bit(pel_t* p_dst, const pel_t* p_src, int stride,
    int lcu_pix_x, int lcu_pix_y, int lcu_width, int lcu_height,
    int* alf_coeff, int b_top_avail, int b_down_avail)
{
    pel_t *imgPad1, *imgPad2, *imgPad3, *imgPad4, *imgPad5, *imgPad6;

    __m256i T00, T01, T10, T11, T20, T21, T30, T31, T40, T41;
    __m256i E00, E01, E10, E11, E20, E21, E30, E31, E40, E41;
    __m256i C0, C1, C2, C3, C4, C5, C6, C7, C8;
    __m256i S00, S01, S10, S11, S20, S21, S30, S31, S40, S41, S50, S51, S60, S61, SS1, SS2, S, S70, S71, S80, S81;
    __m256i mAddOffset;
    __m256i mask;
    __m256i zero = _mm256_setzero_si256();
    int max_pixel = (1 << g_bit_depth) - 1;
    __m256i max_val = _mm256_set1_epi16(max_pixel);

    int i, j;
    int startPos = b_top_avail ? (lcu_pix_y - 4) : lcu_pix_y;
    int endPos = b_down_avail ? (lcu_pix_y + lcu_height - 4) : (lcu_pix_y + lcu_height);
    int xPosEnd = lcu_pix_x + lcu_width;

    p_src += (startPos*stride);
    p_dst += (startPos*stride);

    C0 = _mm256_set1_epi16((pel_t)alf_coeff[0]);
    C1 = _mm256_set1_epi16((pel_t)alf_coeff[1]);
    C2 = _mm256_set1_epi16((pel_t)alf_coeff[2]);
    C3 = _mm256_set1_epi16((pel_t)alf_coeff[3]);
    C4 = _mm256_set1_epi16((pel_t)alf_coeff[4]);
    C5 = _mm256_set1_epi16((pel_t)alf_coeff[5]);
    C6 = _mm256_set1_epi16((pel_t)alf_coeff[6]);
    C7 = _mm256_set1_epi16((pel_t)alf_coeff[7]);
    C8 = _mm256_set1_epi16((pel_t)alf_coeff[8]);

    mAddOffset = _mm256_set1_epi32(32);

    if (lcu_width & 15){
        int xPosEnd15 = xPosEnd - (lcu_width & 15);
        mask = _mm256_loadu_si256((__m256i*)(intrinsic_mask[(lcu_width & 15) - 1]));
        for (i = startPos; i < endPos; i++) {
            int yUp = DAVS2_CLIP3(startPos, endPos - 1, i - 1);
            int yBottom = DAVS2_CLIP3(startPos, endPos - 1, i + 1);
            imgPad1 = p_src + (yBottom - i)*stride;
            imgPad2 = p_src + (yUp - i)*stride;

            yUp = DAVS2_CLIP3(startPos, endPos - 1, i - 2);
            yBottom = DAVS2_CLIP3(startPos, endPos - 1, i + 2);
            imgPad3 = p_src + (yBottom - i)*stride;
            imgPad4 = p_src + (yUp - i)*stride;

            yUp = DAVS2_CLIP3(startPos, endPos - 1, i - 3);
            yBottom = DAVS2_CLIP3(startPos, endPos - 1, i + 3);
            imgPad5 = p_src + (yBottom - i)*stride;
            imgPad6 = p_src + (yUp - i)*stride;

            for (j = lcu_pix_x; j < xPosEnd; j += 16) {
                T00 = _mm256_loadu_si256((__m256i*)&imgPad6[j]);
                T01 = _mm256_loadu_si256((__m256i*)&imgPad5[j]);
                E00 = _mm256_unpacklo_epi16(T00, T01);
                E01 = _mm256_unpackhi_epi16(T00, T01);
                S00 = _mm256_madd_epi16(E00, C0);//前8个像素所有C0*P0的结果
                S01 = _mm256_madd_epi16(E01, C0);//后8个像素所有C0*P0的结果

                T10 = _mm256_loadu_si256((__m256i*)&imgPad4[j]);
                T11 = _mm256_loadu_si256((__m256i*)&imgPad3[j]);
                E10 = _mm256_unpacklo_epi16(T10, T11);
                E11 = _mm256_unpackhi_epi16(T10, T11);
                S10 = _mm256_madd_epi16(E10, C1);//前8个像素所有C1*P1的结果
                S11 = _mm256_madd_epi16(E11, C1);//后8个像素所有C1*P1的结果

                T20 = _mm256_loadu_si256((__m256i*)&imgPad2[j - 1]);
                T21 = _mm256_loadu_si256((__m256i*)&imgPad1[j + 1]);
                E20 = _mm256_unpacklo_epi16(T20, T21);
                E21 = _mm256_unpackhi_epi16(T20, T21);
                S20 = _mm256_madd_epi16(E20, C2);
                S21 = _mm256_madd_epi16(E21, C2);

                T30 = _mm256_loadu_si256((__m256i*)&imgPad2[j]);
                T31 = _mm256_loadu_si256((__m256i*)&imgPad1[j]);
                E30 = _mm256_unpacklo_epi16(T30, T31);
                E31 = _mm256_unpackhi_epi16(T30, T31);
                S30 = _mm256_madd_epi16(E30, C3);
                S31 = _mm256_madd_epi16(E31, C3);

                T40 = _mm256_loadu_si256((__m256i*)&imgPad2[j + 1]);
                T41 = _mm256_loadu_si256((__m256i*)&imgPad1[j - 1]);
                E40 = _mm256_unpacklo_epi16(T40, T41);
                E41 = _mm256_unpackhi_epi16(T40, T41);
                S40 = _mm256_madd_epi16(E40, C4);
                S41 = _mm256_madd_epi16(E41, C4);

                T40 = _mm256_loadu_si256((__m256i*)&p_src[j - 3]);
                T41 = _mm256_loadu_si256((__m256i*)&p_src[j + 3]);
                E40 = _mm256_unpacklo_epi16(T40, T41);
                E41 = _mm256_unpackhi_epi16(T40, T41);
                S50 = _mm256_madd_epi16(E40, C5);
                S51 = _mm256_madd_epi16(E41, C5);

                T40 = _mm256_loadu_si256((__m256i*)&p_src[j - 2]);
                T41 = _mm256_loadu_si256((__m256i*)&p_src[j + 2]);
                E40 = _mm256_unpacklo_epi16(T40, T41);
                E41 = _mm256_unpackhi_epi16(T40, T41);
                S60 = _mm256_madd_epi16(E40, C6);
                S61 = _mm256_madd_epi16(E41, C6);

                T40 = _mm256_loadu_si256((__m256i*)&p_src[j - 1]);
                T41 = _mm256_loadu_si256((__m256i*)&p_src[j + 1]);
                E40 = _mm256_unpacklo_epi16(T40, T41);
                E41 = _mm256_unpackhi_epi16(T40, T41);
                S70 = _mm256_madd_epi16(E40, C7);
                S71 = _mm256_madd_epi16(E41, C7);

                T40 = _mm256_loadu_si256((__m256i*)&p_src[j]);
                E40 = _mm256_unpacklo_epi16(T40, zero);
                E41 = _mm256_unpackhi_epi16(T40, zero);
                S80 = _mm256_madd_epi16(E40, C8);
                S81 = _mm256_madd_epi16(E41, C8);

                SS1 = _mm256_add_epi32(S00, S10);
                SS1 = _mm256_add_epi32(SS1, S20);
                SS1 = _mm256_add_epi32(SS1, S30);
                SS1 = _mm256_add_epi32(SS1, S40);
                SS1 = _mm256_add_epi32(SS1, S50);
                SS1 = _mm256_add_epi32(SS1, S60);
                SS1 = _mm256_add_epi32(SS1, S70);
                SS1 = _mm256_add_epi32(SS1, S80);

                SS2 = _mm256_add_epi32(S01, S11);
                SS2 = _mm256_add_epi32(SS2, S21);
                SS2 = _mm256_add_epi32(SS2, S31);
                SS2 = _mm256_add_epi32(SS2, S41);
                SS2 = _mm256_add_epi32(SS2, S51);
                SS2 = _mm256_add_epi32(SS2, S61);
                SS2 = _mm256_add_epi32(SS2, S71);
                SS2 = _mm256_add_epi32(SS2, S81);

                SS1 = _mm256_add_epi32(SS1, mAddOffset);
                SS1 = _mm256_srai_epi32(SS1, 6);

                SS2 = _mm256_add_epi32(SS2, mAddOffset);
                SS2 = _mm256_srai_epi32(SS2, 6);

                S = _mm256_packus_epi32(SS1, SS2);
                S = _mm256_min_epu16(S, max_val);
                if (j != xPosEnd15){
                    _mm256_storeu_si256((__m256i*)(p_dst + j), S);
                }
                else{
                    _mm256_maskstore_epi32((int *)(p_dst + j), mask, S);
                    break;
                }
            }

            p_src += stride;
            p_dst += stride;
        }
    }
    else{
        for (i = startPos; i < endPos; i++) {
            int yUp = DAVS2_CLIP3(startPos, endPos - 1, i - 1);
            int yBottom = DAVS2_CLIP3(startPos, endPos - 1, i + 1);
            imgPad1 = p_src + (yBottom - i)*stride;
            imgPad2 = p_src + (yUp - i)*stride;

            yUp = DAVS2_CLIP3(startPos, endPos - 1, i - 2);
            yBottom = DAVS2_CLIP3(startPos, endPos - 1, i + 2);
            imgPad3 = p_src + (yBottom - i)*stride;
            imgPad4 = p_src + (yUp - i)*stride;

            yUp = DAVS2_CLIP3(startPos, endPos - 1, i - 3);
            yBottom = DAVS2_CLIP3(startPos, endPos - 1, i + 3);
            imgPad5 = p_src + (yBottom - i)*stride;
            imgPad6 = p_src + (yUp - i)*stride;

            for (j = lcu_pix_x; j < xPosEnd; j += 16) {
                T00 = _mm256_loadu_si256((__m256i*)&imgPad6[j]);
                T01 = _mm256_loadu_si256((__m256i*)&imgPad5[j]);
                E00 = _mm256_unpacklo_epi16(T00, T01);
                E01 = _mm256_unpackhi_epi16(T00, T01);
                S00 = _mm256_madd_epi16(E00, C0);//前8个像素所有C0*P0的结果
                S01 = _mm256_madd_epi16(E01, C0);//后8个像素所有C0*P0的结果

                T10 = _mm256_loadu_si256((__m256i*)&imgPad4[j]);
                T11 = _mm256_loadu_si256((__m256i*)&imgPad3[j]);
                E10 = _mm256_unpacklo_epi16(T10, T11);
                E11 = _mm256_unpackhi_epi16(T10, T11);
                S10 = _mm256_madd_epi16(E10, C1);//前8个像素所有C1*P1的结果
                S11 = _mm256_madd_epi16(E11, C1);//后8个像素所有C1*P1的结果

                T20 = _mm256_loadu_si256((__m256i*)&imgPad2[j - 1]);
                T21 = _mm256_loadu_si256((__m256i*)&imgPad1[j + 1]);
                E20 = _mm256_unpacklo_epi16(T20, T21);
                E21 = _mm256_unpackhi_epi16(T20, T21);
                S20 = _mm256_madd_epi16(E20, C2);
                S21 = _mm256_madd_epi16(E21, C2);

                T30 = _mm256_loadu_si256((__m256i*)&imgPad2[j]);
                T31 = _mm256_loadu_si256((__m256i*)&imgPad1[j]);
                E30 = _mm256_unpacklo_epi16(T30, T31);
                E31 = _mm256_unpackhi_epi16(T30, T31);
                S30 = _mm256_madd_epi16(E30, C3);
                S31 = _mm256_madd_epi16(E31, C3);

                T40 = _mm256_loadu_si256((__m256i*)&imgPad2[j + 1]);
                T41 = _mm256_loadu_si256((__m256i*)&imgPad1[j - 1]);
                E40 = _mm256_unpacklo_epi16(T40, T41);
                E41 = _mm256_unpackhi_epi16(T40, T41);
                S40 = _mm256_madd_epi16(E40, C4);
                S41 = _mm256_madd_epi16(E41, C4);

                T40 = _mm256_loadu_si256((__m256i*)&p_src[j - 3]);
                T41 = _mm256_loadu_si256((__m256i*)&p_src[j + 3]);
                E40 = _mm256_unpacklo_epi16(T40, T41);
                E41 = _mm256_unpackhi_epi16(T40, T41);
                S50 = _mm256_madd_epi16(E40, C5);
                S51 = _mm256_madd_epi16(E41, C5);

                T40 = _mm256_loadu_si256((__m256i*)&p_src[j - 2]);
                T41 = _mm256_loadu_si256((__m256i*)&p_src[j + 2]);
                E40 = _mm256_unpacklo_epi16(T40, T41);
                E41 = _mm256_unpackhi_epi16(T40, T41);
                S60 = _mm256_madd_epi16(E40, C6);
                S61 = _mm256_madd_epi16(E41, C6);

                T40 = _mm256_loadu_si256((__m256i*)&p_src[j - 1]);
                T41 = _mm256_loadu_si256((__m256i*)&p_src[j + 1]);
                E40 = _mm256_unpacklo_epi16(T40, T41);
                E41 = _mm256_unpackhi_epi16(T40, T41);
                S70 = _mm256_madd_epi16(E40, C7);
                S71 = _mm256_madd_epi16(E41, C7);

                T40 = _mm256_loadu_si256((__m256i*)&p_src[j]);
                E40 = _mm256_unpacklo_epi16(T40, zero);
                E41 = _mm256_unpackhi_epi16(T40, zero);
                S80 = _mm256_madd_epi16(E40, C8);
                S81 = _mm256_madd_epi16(E41, C8);

                SS1 = _mm256_add_epi32(S00, S10);
                SS1 = _mm256_add_epi32(SS1, S20);
                SS1 = _mm256_add_epi32(SS1, S30);
                SS1 = _mm256_add_epi32(SS1, S40);
                SS1 = _mm256_add_epi32(SS1, S50);
                SS1 = _mm256_add_epi32(SS1, S60);
                SS1 = _mm256_add_epi32(SS1, S70);
                SS1 = _mm256_add_epi32(SS1, S80);

                SS2 = _mm256_add_epi32(S01, S11);
                SS2 = _mm256_add_epi32(SS2, S21);
                SS2 = _mm256_add_epi32(SS2, S31);
                SS2 = _mm256_add_epi32(SS2, S41);
                SS2 = _mm256_add_epi32(SS2, S51);
                SS2 = _mm256_add_epi32(SS2, S61);
                SS2 = _mm256_add_epi32(SS2, S71);
                SS2 = _mm256_add_epi32(SS2, S81);

                SS1 = _mm256_add_epi32(SS1, mAddOffset);
                SS1 = _mm256_srai_epi32(SS1, 6);

                SS2 = _mm256_add_epi32(SS2, mAddOffset);
                SS2 = _mm256_srai_epi32(SS2, 6);

                S = _mm256_packus_epi32(SS1, SS2);
                S = _mm256_min_epu16(S, max_val);

                _mm256_storeu_si256((__m256i*)(p_dst + j), S);

            }

            p_src += stride;
            p_dst += stride;
        }
    }
}

#endif
