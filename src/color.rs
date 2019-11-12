use num_traits::{NumCast, ToPrimitive, Zero};
use std::ops::{Index, IndexMut};

use buffer::Pixel;
use traits::Primitive;

/// An enumeration over supported color types and bit depths
#[derive(Copy, PartialEq, Eq, Debug, Clone, Hash)]
pub enum ColorType {
    /// Pixel is 8-bit luminance
    L8,
    /// Pixel is 8-bit luminance with an alpha channel
    La8,
    /// Pixel contains 8-bit R, G and B channels
    Rgb8,
    /// Pixel is 8-bit RGB with an alpha channel
    Rgba8,

    /// Pixel is 16-bit luminance
    L16,
    /// Pixel is 16-bit luminance with an alpha channel
    La16,
    /// Pixel is 16-bit RGB
    Rgb16,
    /// Pixel is 16-bit RGBA
    Rgba16,

    /// Pixel contains 8-bit B, G and R channels
    Bgr8,
    /// Pixel is 8-bit BGR with an alpha channel
    Bgra8,

    #[doc(hidden)]
    __Nonexhaustive,
}

impl ColorType {
    /// Returns the number of bytes contained in a pixel of `ColorType` ```c```
    pub fn bytes_per_pixel(self) -> u8 {
        match self {
            ColorType::L8 => 1,
            ColorType::L16 | ColorType::La8 => 2,
            ColorType::Rgb8 | ColorType::Bgr8 => 3,
            ColorType::Rgba8 | ColorType::Bgra8 | ColorType::La16 => 4,
            ColorType::Rgb16 => 6,
            ColorType::Rgba16 => 8,
            ColorType::__Nonexhaustive => unreachable!(),
        }
    }

    /// Returns the number of bits contained in a pixel of `ColorType` ```c``` (which will always be
    /// a multiple of 8).
    pub fn bits_per_pixel(self) -> u16 {
        <u16 as From<u8>>::from(self.bytes_per_pixel()) * 8
    }

    /// Returns the number of color channels that make up this pixel
    pub fn channel_count(self) -> u8 {
        let e: ExtendedColorType = self.into();
        e.channel_count()
    }
}

#[derive(Copy, PartialEq, Eq, Debug, Clone, Hash)]
pub enum ExtendedColorType {
    L1,
    La1,
    Rgb1,
    Rgba1,
    L2,
    La2,
    Rgb2,
    Rgba2,
    L4,
    La4,
    Rgb4,
    Rgba4,
    L8,
    La8,
    Rgb8,
    Rgba8,
    L16,
    La16,
    Rgb16,
    Rgba16,
    Bgr8,
    Bgra8,

    /// Pixel is of unknown color type with the specified bits per pixel. This can apply to pixels
    /// which are associated with an external palette. In that case, the pixel value is an index
    /// into the palette.
    Unknown(u8),

    #[doc(hidden)]
    __Nonexhaustive,
}

impl ExtendedColorType {
    pub fn channel_count(self) -> u8 {
        match self {
            ExtendedColorType::L1 |
            ExtendedColorType::L2 |
            ExtendedColorType::L4 |
            ExtendedColorType::L8 |
            ExtendedColorType::L16 |
            ExtendedColorType::Unknown(_) => 1,
            ExtendedColorType::La1 |
            ExtendedColorType::La2 |
            ExtendedColorType::La4 |
            ExtendedColorType::La8 |
            ExtendedColorType::La16 => 2,
            ExtendedColorType::Rgb1 |
            ExtendedColorType::Rgb2 |
            ExtendedColorType::Rgb4 |
            ExtendedColorType::Rgb8 |
            ExtendedColorType::Rgb16 |
            ExtendedColorType::Bgr8 => 3,
            ExtendedColorType::Rgba1 |
            ExtendedColorType::Rgba2 |
            ExtendedColorType::Rgba4 |
            ExtendedColorType::Rgba8 |
            ExtendedColorType::Rgba16 |
            ExtendedColorType::Bgra8 => 4,
            ExtendedColorType::__Nonexhaustive => unreachable!(),
        }
    }
}
impl From<ColorType> for ExtendedColorType {
    fn from(c: ColorType) -> Self {
        match c {
            ColorType::L8 => ExtendedColorType::L8,
            ColorType::La8 => ExtendedColorType::La8,
            ColorType::Rgb8 => ExtendedColorType::Rgb8,
            ColorType::Rgba8 => ExtendedColorType::Rgba8,
            ColorType::L16 => ExtendedColorType::L16,
            ColorType::La16 => ExtendedColorType::La16,
            ColorType::Rgb16 => ExtendedColorType::Rgb16,
            ColorType::Rgba16 => ExtendedColorType::Rgba16,
            ColorType::Bgr8 => ExtendedColorType::Bgr8,
            ColorType::Bgra8 => ExtendedColorType::Bgra8,
            ColorType::__Nonexhaustive => unreachable!(),
        }
    }
}

macro_rules! define_colors {
    {$(
        $ident:ident,
        $channels: expr,
        $alphas: expr,
        $interpretation: expr,
        $color_type: expr,
        #[$doc:meta];
    )*} => {

$( // START Structure definitions

#[$doc]
#[derive(PartialEq, Eq, Clone, Debug, Copy, Hash)]
#[repr(C)]
#[allow(missing_docs)]
pub struct $ident<T: Primitive> (pub [T; $channels]);

impl<T: Primitive + 'static> Pixel for $ident<T> {
    type Subpixel = T;

    const CHANNEL_COUNT: u8 = $channels;

    const COLOR_MODEL: &'static str = $interpretation;

    const COLOR_TYPE: ColorType = $color_type;

    #[inline(always)]
    fn channels(&self) -> &[T] {
        &self.0
    }
    #[inline(always)]
    fn channels_mut(&mut self) -> &mut [T] {
        &mut self.0
    }

    fn channels4(&self) -> (T, T, T, T) {
        const CHANNELS: usize = $channels;
        let mut channels = [T::max_value(); 4];
        channels[0..CHANNELS].copy_from_slice(&self.0);
        (channels[0], channels[1], channels[2], channels[3])
    }

    fn from_channels(a: T, b: T, c: T, d: T,) -> $ident<T> {
        const CHANNELS: usize = $channels;
        *<$ident<T> as Pixel>::from_slice(&[a, b, c, d][..CHANNELS])
    }

    fn from_slice(slice: &[T]) -> &$ident<T> {
        assert_eq!(slice.len(), $channels);
        unsafe { &*(slice.as_ptr() as *const $ident<T>) }
    }
    fn from_slice_mut(slice: &mut [T]) -> &mut $ident<T> {
        assert_eq!(slice.len(), $channels);
        unsafe { &mut *(slice.as_ptr() as *mut $ident<T>) }
    }

    fn to_rgb(&self) -> Rgb<T> {
        let mut pix = Rgb([Zero::zero(), Zero::zero(), Zero::zero()]);
        pix.from_color(self);
        pix
    }

    fn to_bgr(&self) -> Bgr<T> {
        let mut pix = Bgr([Zero::zero(), Zero::zero(), Zero::zero()]);
        pix.from_color(self);
        pix
    }

    fn to_rgba(&self) -> Rgba<T> {
        let mut pix = Rgba([Zero::zero(), Zero::zero(), Zero::zero(), Zero::zero()]);
        pix.from_color(self);
        pix
    }

    fn to_bgra(&self) -> Bgra<T> {
        let mut pix = Bgra([Zero::zero(), Zero::zero(), Zero::zero(), Zero::zero()]);
        pix.from_color(self);
        pix
    }

    fn to_luma(&self) -> Luma<T> {
        let mut pix = Luma([Zero::zero()]);
        pix.from_color(self);
        pix
    }

    fn to_luma_alpha(&self) -> LumaA<T> {
        let mut pix = LumaA([Zero::zero(), Zero::zero()]);
        pix.from_color(self);
        pix
    }

    fn map<F>(& self, f: F) -> $ident<T> where F: FnMut(T) -> T {
        let mut this = (*self).clone();
        this.apply(f);
        this
    }

    fn apply<F>(&mut self, mut f: F) where F: FnMut(T) -> T {
        for v in &mut self.0 {
            *v = f(*v)
        }
    }

    fn map_with_alpha<F, G>(&self, f: F, g: G) -> $ident<T> where F: FnMut(T) -> T, G: FnMut(T) -> T {
        let mut this = (*self).clone();
        this.apply_with_alpha(f, g);
        this
    }

    fn apply_with_alpha<F, G>(&mut self, mut f: F, mut g: G) where F: FnMut(T) -> T, G: FnMut(T) -> T {
        const ALPHA: usize = $channels - $alphas;
        for v in self.0[..ALPHA].iter_mut() {
            *v = f(*v)
        }
        // The branch of this match is `const`. This way ensures that no subexpression fails the
        // `const_err` lint (the expression `self.0[ALPHA]` would).
        if let Some(v) = self.0.get_mut(ALPHA) {
            *v = g(*v)
        }
    }

    fn map2<F>(&self, other: &Self, f: F) -> $ident<T> where F: FnMut(T, T) -> T {
        let mut this = (*self).clone();
        this.apply2(other, f);
        this
    }

    fn apply2<F>(&mut self, other: &$ident<T>, mut f: F) where F: FnMut(T, T) -> T {
        for (a, &b) in self.0.iter_mut().zip(other.0.iter()) {
            *a = f(*a, b)
        }
    }

    fn invert(&mut self) {
        Invert::invert(self)
    }

    fn blend(&mut self, other: &$ident<T>) {
        Blend::blend(self, other)
    }
}

impl<T: Primitive> Index<usize> for $ident<T> {
    type Output = T;
    #[inline(always)]
    fn index(&self, _index: usize) -> &T {
        &self.0[_index]
    }
}

impl<T: Primitive> IndexMut<usize> for $ident<T> {
    #[inline(always)]
    fn index_mut(&mut self, _index: usize) -> &mut T {
        &mut self.0[_index]
    }
}

)* // END Structure definitions

    }
}

define_colors! {
    Rgb, 3, 0, "RGB", ColorType::Rgb8, #[doc = "RGB colors"];
    Bgr, 3, 0, "BGR", ColorType::Bgr8, #[doc = "BGR colors"];
    Luma, 1, 0, "Y", ColorType::L8, #[doc = "Grayscale colors"];
    Rgba, 4, 1, "RGBA", ColorType::Rgba8, #[doc = "RGB colors + alpha channel"];
    Bgra, 4, 1, "BGRA", ColorType::Bgra8, #[doc = "BGR colors + alpha channel"];
    LumaA, 2, 1, "YA", ColorType::La8, #[doc = "Grayscale colors + alpha channel"];

    Rgb16, 3, 0, "RGB", ColorType::Rgb16, #[doc = "RGB 16-bpc colors"];
    Luma16, 1, 0, "Y", ColorType::L16, #[doc = "Grayscale 16-bpc colors"];
    Rgba16, 4, 1, "RGBA", ColorType::Rgba16, #[doc = "RGB 16-bpc colors + alpha channel"];
    LumaA16, 2, 1, "YA", ColorType::La16, #[doc = "Grayscale 16-bpc colors + alpha channel"];
}

// /// Private trait for deduplicating common logic for 8-bpc and 16-bpc colors.
// trait ColorGroup<T: Primitive> {
//     type Luma: Pixel<Subpixel=T>;
//     type LumaA: Pixel<Subpixel=T>;
//     type Rgb: Pixel<Subpixel=T>;
//     type Rgba: Pixel<Subpixel=T>;
// }

// /// Private type for common 8-bpc colors.
// struct ColorGroup8<T: Primitive>;
// impl<T: Primitive> ColorGroup<T> for ColorGroup8<T> {
//     type Luma = Luma<T>;
//     type LumaA = LumaA<T>;
//     type Rgb = Rgb<T>;
//     type Rgba = Rgba<T>;
// }

// /// Private type for common 16-bpc colors.
// struct ColorGroup16<T: Primitive>;
// impl<T: Primitive> ColorGroup<T> for ColorGroup16<T> {
//     type Luma = Luma16<T>;
//     type LumaA = LumaA16<T>;
//     type Rgb = Rgb16<T>;
//     type Rgba = Rgba16<T>;
// }

/// Private trait for deduplicating common logic for 8-bpc and 16-bpc luma colors.
// trait LumaLike<T: Primitive>: Pixel<Subpixel=T> {
//     type Group: ColorGroup<T>;
// }
// impl<T: Primitive> LumaLike<T> for Luma<T> {
//     type Group = ColorGroup8<T>;
// }
// impl<T: Primitive> LumaLike<T> for Luma16<T> {
//     type Group = ColorGroup16<T>;
// }

/// Provides color conversions for the different pixel types.
pub trait FromColor<Other> {
    /// Changes `self` to represent `Other` in the color space of `Self`
    fn from_color(&mut self, &Other);
}

// Self->Self: just copy
impl<A: Copy> FromColor<A> for A {
    fn from_color(&mut self, other: &A) {
        *self = *other;
    }
}

/// Coefficients to transform from sRGB to a CIE Y (luminance) value.
const sRGB_luma: [f32; 3] = [0.2126, 0.7152, 0.0722];

#[inline]
const fn rgb_to_luma<T: Primitive>(rgb: &[T]) -> T {
    let l = sRGB_luma[0] * rgb[0].to_f32().unwrap()
        + sRGB_luma[1] * rgb[1].to_f32().unwrap()
        + sRGB_luma[2] * rgb[2].to_f32().unwrap();
    NumCast::from(l).unwrap()
}

#[inline]
const fn bgr_to_luma<T: Primitive>(bgr: &[T]) -> T {
    let l = sRGB_luma[0] * bgr[2].to_f32().unwrap()
        + sRGB_luma[1] * bgr[1].to_f32().unwrap()
        + sRGB_luma[2] * bgr[0].to_f32().unwrap();
    NumCast::from(l).unwrap()
}

#[inline]
const fn downcast_channel<T16: Primitive + 'static, T8: Primitive + 'static>(c16: T16) -> T8 {
    NumCast::from(c16.to_u32().unwrap() / 2).unwrap()
}

macro_rules! bit_depth_common {
    ($luma:ident, $luma_a:ident, $rgb:ident, $rgba:ident) => {
        // `FromColor` for Luma

        impl<T: Primitive + 'static> FromColor<$luma_a<T>> for $luma<T> {
            fn from_color(&mut self, other: &$luma_a<T>) {
                self.channels_mut()[0] = other.channels()[0]
            }
        }

        impl<T: Primitive + 'static> FromColor<$rgb<T>> for $luma<T> {
            fn from_color(&mut self, other: &$rgb<T>) {
                let gray = self.channels_mut();
                gray[0] = rgb_to_luma(other.channels());
            }
        }

        impl<T: Primitive + 'static> FromColor<$rgba<T>> for $luma<T> {
            fn from_color(&mut self, other: &$rgba<T>) {
                let gray = self.channels_mut();
                gray[0] = rgb_to_luma(other.channels());
            }
        }

        // `FromColor` for LumaA

        impl<T: Primitive + 'static> FromColor<$luma<T>> for $luma_a<T> {
            fn from_color(&mut self, other: &$luma<T>) {
                self.channels_mut()[0] = other.channels()[0];
                // TODO
            }
        }

        impl<T: Primitive + 'static> FromColor<$rgb<T>> for $luma_a<T> {
            fn from_color(&mut self, other: &$rgb<T>) {
                let gray_a = self.channels_mut();
                let rgb = other.channels();
                gray_a[0] = rgb_to_luma(rgb);
                gray_a[1] = T::max_value();
            }
        }

        impl<T: Primitive + 'static> FromColor<$rgba<T>> for $luma_a<T> {
            fn from_color(&mut self, other: &$rgba<T>) {
                let gray_a = self.channels_mut();
                let rgba = other.channels();
                gray_a[0] = rgb_to_luma(rgba);
                gray_a[1] = rgba[3];
            }
        }

        // `FromColor` for Rgb

        impl<T: Primitive + 'static> FromColor<$luma<T>> for $rgb<T> {
            fn from_color(&mut self, gray: &$luma<T>) {
                let rgb = self.channels_mut();
                let gray = gray.channels()[0];
                rgb[0] = gray;
                rgb[1] = gray;
                rgb[2] = gray;
            }
        }

        impl<T: Primitive + 'static> FromColor<$luma_a<T>> for $rgb<T> {
            fn from_color(&mut self, other: &$luma_a<T>) {
                let rgb = self.channels_mut();
                let gray = other.channels()[0];
                rgb[0] = gray;
                rgb[1] = gray;
                rgb[2] = gray;
            }
        }

        impl<T: Primitive + 'static> FromColor<$rgba<T>> for $rgb<T> {
            fn from_color(&mut self, other: &$rgba<T>) {
                let rgb = self.channels_mut();
                let rgba = other.channels();
                rgb[0] = rgba[0];
                rgb[1] = rgba[1];
                rgb[2] = rgba[2];
            }
        }

        // `FromColor` for Rgba 

        impl<T: Primitive + 'static> FromColor<$luma<T>> for $rgba<T> {
            fn from_color(&mut self, gray: &$luma<T>) {
                let rgba = self.channels_mut();
                let gray = gray.channels()[0];
                rgba[0] = gray;
                rgba[1] = gray;
                rgba[2] = gray;
                rgba[3] = T::max_value();
            }
        }

        impl<T: Primitive + 'static> FromColor<$luma_a<T>> for $rgba<T> {
            fn from_color(&mut self, other: &$luma_a<T>) {
                let rgba = self.channels_mut();
                let &[gray, alpha] = other.channels();
                rgba[0] = gray;
                rgba[1] = gray;
                rgba[2] = gray;
                rgba[3] = alpha;
            }
        }

        impl<T: Primitive + 'static> FromColor<$rgb<T>> for $rgba<T> {
            fn from_color(&mut self, other: &$rgb<T>) {
                let rgba = self.channels_mut();
                let rgb = other.channels();
                rgba[0] = rgb[0];
                rgba[1] = rgb[1];
                rgba[2] = rgb[2];
                rgba[3] = T::max_value();
            }
        }
    };
}

bit_depth_common!(Luma, LumaA, Rgb, Rgba);
bit_depth_common!(Luma16, LumaA16, Rgb16, Rgba16);

macro_rules! downcast_bit_depth_early {
    ($src:ident, $intermediate:ident, $channels:expr, $dst:ident) => {
        impl<T16: Primitive + 'static, T8: Primitive + 'static> FromColor<$src<T16>> for $dst<T8> {
            fn from_color(&mut self, other: &$src<T16>) {
                let mut intermediate: $intermediate<T8> = $intermediate([Zero::zero(); $channels]);
                intermediate.from_color(other);
                self.from_color(&intermediate);
            }
        }
    };
}

// Downcasts
// LumaA
downcast_bit_depth_early!(Luma16, Luma, 1, LumaA);
downcast_bit_depth_early!(Rgb16, Rgb, 3, LumaA);
downcast_bit_depth_early!(Rgba16, Rgba, 4, LumaA);
// Rgb
downcast_bit_depth_early!(Luma16, Luma, 1, Rgb);
downcast_bit_depth_early!(LumaA16, LumaA, 2, Rgb);
downcast_bit_depth_early!(Rgba16, Rgba, 4, Rgb);
// Rgba
downcast_bit_depth_early!(Luma16, Luma, 1, Rgba);
downcast_bit_depth_early!(LumaA16, LumaA, 2, Rgba);
downcast_bit_depth_early!(Rgb16, Rgb, 3, Rgba);
// Bgr
downcast_bit_depth_early!(Luma16, Luma, 1, Bgr);
downcast_bit_depth_early!(LumaA16, LumaA, 2, Bgr);
downcast_bit_depth_early!(Rgb16, Rgb, 3, Bgr);
downcast_bit_depth_early!(Rgba16, Rgba, 4, Bgr);
// Bgra
downcast_bit_depth_early!(Luma16, Luma, 1, Bgra);
downcast_bit_depth_early!(LumaA16, LumaA, 2, Bgra);
downcast_bit_depth_early!(Rgb16, Rgb, 3, Bgra);
downcast_bit_depth_early!(Rgba16, Rgba, 4, Bgra);


impl<T: Primitive + 'static> FromColor<Bgra<T>> for Luma<T> {
    fn from_color(&mut self, other: &Bgra<T>) {
        let gray = self.channels_mut();
        let bgra = other.channels();
        gray[0] = bgr_to_luma(bgra);
    }
}

impl<T: Primitive + 'static> FromColor<Bgr<T>> for Luma<T> {
    fn from_color(&mut self, other: &Bgr<T>) {
        let gray = self.channels_mut();
        let bgr = other.channels();
        gray[0] = bgr_to_luma(bgr);
    }
}


impl<T16: Primitive + 'static, T8: Primitive + 'static> FromColor<Rgba16<T16>> for Luma<T8> {
    fn from_color(&mut self, other: &Rgba16<T16>) {
        let gray = self.channels_mut();
        let rgb = other.channels();
        let l = rgb_to_luma(rgb);
        gray[0] = downcast_channel(l);
    }
}

impl<T16: Primitive + 'static, T8: Primitive + 'static> FromColor<Rgb16<T16>> for Luma<T8> {
    fn from_color(&mut self, other: &Rgb16<T16>) {
        let gray = self.channels_mut();
        let rgb = other.channels();
        let l = rgb_to_luma(rgb);
        gray[0] = downcast_channel(l);
    }
}

impl<T16: Primitive + 'static, T8: Primitive + 'static> FromColor<Luma16<T16>> for Luma<T8> {
    fn from_color(&mut self, other: &Luma16<T16>) {
        let l = other.channels()[0];
        self.channels_mut()[0] = downcast_channel(l);
    }
}

impl<T16: Primitive + 'static, T8: Primitive + 'static> FromColor<LumaA16<T16>> for Luma<T8> {
    fn from_color(&mut self, other: &LumaA16<T16>) {
        let l = other.channels()[0];
        self.channels_mut()[0] = downcast_channel(l);
    }
}

// `FromColor` for LumaA

impl<T: Primitive + 'static> FromColor<Bgra<T>> for LumaA<T> {
    fn from_color(&mut self, other: &Bgra<T>) {
        let gray_a = self.channels_mut();
        let bgra = other.channels();
        gray_a[0] = bgr_to_luma(bgra);
        gray_a[1] = bgra[3];
    }
}


impl<T: Primitive + 'static> FromColor<Bgr<T>> for LumaA<T> {
    fn from_color(&mut self, other: &Bgr<T>) {
        let gray_a = self.channels_mut();
        let bgr = other.channels();
        gray_a[0] = bgr_to_luma(bgr);
        gray_a[1] = T::max_value();
    }
}

// `FromColor` for RGBA

impl<T: Primitive + 'static> FromColor<Bgr<T>> for Rgba<T> {
    fn from_color(&mut self, other: &Bgr<T>) {
        let rgba = self.channels_mut();
        let bgr = other.channels();
        rgba[0] = bgr[2];
        rgba[1] = bgr[1];
        rgba[2] = bgr[0];
        rgba[3] = T::max_value();
    }
}

impl<T: Primitive + 'static> FromColor<Bgra<T>> for Rgba<T> {
    fn from_color(&mut self, other: &Bgra<T>) {
        let rgba = self.channels_mut();
        let bgra = other.channels();
        rgba[0] = bgra[2];
        rgba[1] = bgra[1];
        rgba[2] = bgra[0];
        rgba[3] = bgra[3];
    }
}

impl<T16: Primitive + 'static, T8: Primitive + 'static> FromColor<Rgba16<T16>> for Rgba<T8> {
    fn from_color(&mut self, other: &Rgba16<T16>) {
        let rgba = self.channels_mut();
        let gray = downcast_channel(other.channels()[0]);
        rgba[0] = gray;
        rgba[1] = gray;
        rgba[2] = gray;
        rgba[3] = T8::max_value();
    }
}

// impl<T16: Primitive + 'static, T8: Primitive + 'static> FromColor<LumaA16<T16>> for Rgba<T8> {
//     fn from_color(&mut self, other: &LumaA16<T16>) {
//         let rgba = self.channels_mut();
//         let gray = downcast_channel(other.channels()[0]);
//         rgba[0] = gray;
//         rgba[1] = gray;
//         rgba[2] = gray;
//         rgba[3] = T8::max_value();
//     }
// }

// `FromColor` for BGRA

impl<T: Primitive + 'static> FromColor<Rgb<T>> for Bgra<T> {
    fn from_color(&mut self, other: &Rgb<T>) {
        let bgra = self.channels_mut();
        let rgb = other.channels();
        bgra[0] = rgb[2];
        bgra[1] = rgb[1];
        bgra[2] = rgb[0];
        bgra[3] = T::max_value();
    }
}


impl<T: Primitive + 'static> FromColor<Bgr<T>> for Bgra<T> {
    fn from_color(&mut self, other: &Bgr<T>) {
        let bgra = self.channels_mut();
        let bgr = other.channels();
        bgra[0] = bgr[0];
        bgra[1] = bgr[1];
        bgra[2] = bgr[2];
        bgra[3] = T::max_value();
    }
}


impl<T: Primitive + 'static> FromColor<Rgba<T>> for Bgra<T> {
    fn from_color(&mut self, other: &Rgba<T>) {
        let bgra = self.channels_mut();
        let rgba = other.channels();
        bgra[2] = rgba[0];
        bgra[1] = rgba[1];
        bgra[0] = rgba[2];
        bgra[3] = rgba[3];
    }
}

impl<T: Primitive + 'static> FromColor<LumaA<T>> for Bgra<T> {
    fn from_color(&mut self, other: &LumaA<T>) {
        let bgra = self.channels_mut();
        let gray = other.channels();
        bgra[0] = gray[0];
        bgra[1] = gray[0];
        bgra[2] = gray[0];
        bgra[3] = gray[1];
    }
}

impl<T: Primitive + 'static> FromColor<Luma<T>> for Bgra<T> {
    fn from_color(&mut self, gray: &Luma<T>) {
        let bgra = self.channels_mut();
        let gray = gray.channels()[0];
        bgra[0] = gray;
        bgra[1] = gray;
        bgra[2] = gray;
        bgra[3] = T::max_value();
    }
}

/// `FromColor` for RGB

impl<T: Primitive + 'static> FromColor<Bgra<T>> for Rgb<T> {
    fn from_color(&mut self, other: &Bgra<T>) {
        let rgb = self.channels_mut();
        let bgra = other.channels();
        rgb[0] = bgra[2];
        rgb[1] = bgra[1];
        rgb[2] = bgra[0];
    }
}

impl<T: Primitive + 'static> FromColor<Bgr<T>> for Rgb<T> {
    fn from_color(&mut self, other: &Bgr<T>) {
        let rgb = self.channels_mut();
        let bgr = other.channels();
        rgb[0] = bgr[2];
        rgb[1] = bgr[1];
        rgb[2] = bgr[0];
    }
}

// impl<T16: Primitive + 'static, T8: Primitive + 'static> FromColor<Rgba16<T16>> for Rgb<T8> {
//     fn from_color(&mut self, other: &Rgba16<T16>) {
//         for (c8, c16) in self.channels_mut().iter_mut().zip(other.channels()) {
//             let c = c16.to_u32().unwrap() / 2;
//             *c8 = NumCast::from(c).unwrap();
//         }
//     }
// }

impl<T16: Primitive + 'static, T8: Primitive + 'static> FromColor<Rgb16<T16>> for Rgb<T8> {
    fn from_color(&mut self, other: &Rgb16<T16>) {
        for (c8, &c16) in self.channels_mut().iter_mut().zip(other.channels()) {
            *c8 = downcast_channel(c16);
        }
    }
}

// impl<T16: Primitive + 'static, T8: Primitive + 'static> FromColor<LumaA16<T16>> for Rgb<T8> {
//     fn from_color(&mut self, gray: &LumaA16<T16>) {
//         let rgba = self.channels_mut();
//         let gray = downcast_channel(gray.channels()[0]);
//         rgba[0] = gray;
//         rgba[1] = gray;
//         rgba[2] = gray;
//     }
// }

/// `FromColor` for BGR

impl<T: Primitive + 'static> FromColor<Rgba<T>> for Bgr<T> {
    fn from_color(&mut self, other: &Rgba<T>) {
        let bgr = self.channels_mut();
        let rgba = other.channels();
        bgr[0] = rgba[2];
        bgr[1] = rgba[1];
        bgr[2] = rgba[0];
    }
}

impl<T: Primitive + 'static> FromColor<Rgb<T>> for Bgr<T> {
    fn from_color(&mut self, other: &Rgb<T>) {
        let bgr = self.channels_mut();
        let rgb = other.channels();
        bgr[0] = rgb[2];
        bgr[1] = rgb[1];
        bgr[2] = rgb[0];
    }
}


impl<T: Primitive + 'static> FromColor<Bgra<T>> for Bgr<T> {
    fn from_color(&mut self, other: &Bgra<T>) {
        let bgr = self.channels_mut();
        let bgra = other.channels();
        bgr[0] = bgra[0];
        bgr[1] = bgra[1];
        bgr[2] = bgra[2];
    }
}

impl<T: Primitive + 'static> FromColor<LumaA<T>> for Bgr<T> {
    fn from_color(&mut self, other: &LumaA<T>) {
        let bgr = self.channels_mut();
        let gray = other.channels()[0];
        bgr[0] = gray;
        bgr[1] = gray;
        bgr[2] = gray;
    }
}

impl<T: Primitive + 'static> FromColor<Luma<T>> for Bgr<T> {
    fn from_color(&mut self, gray: &Luma<T>) {
        let bgr = self.channels_mut();
        let gray = gray.channels()[0];
        bgr[0] = gray;
        bgr[1] = gray;
        bgr[2] = gray;
    }
}


impl<T16: Primitive + 'static, T8: Primitive + 'static> FromColor<LumaA16<T16>> for LumaA<T8> {
    fn from_color(&mut self, other: &LumaA16<T16>) {
        let la8 = self.channels_mut();
        let &[gray, alpha] = other.channels();
        la8[0] = downcast_channel(gray);
        la8[1] = downcast_channel(alpha);
    }
}


/// Blends a color inter another one
pub(crate) trait Blend {
    /// Blends a color in-place.
    fn blend(&mut self, other: &Self);
}

macro_rules! blend_bit_depth_common {
    ($luma:ident, $luma_a:ident, $rgb:ident, $rgba:ident) => {

impl<T: Primitive> Blend for $luma<T> {
    fn blend(&mut self, other: &$luma<T>) {
        *self = *other
    }
}

impl<T: Primitive> Blend for $luma_a<T> {
    fn blend(&mut self, other: &$luma_a<T>) {
        let max_t = T::max_value();
        let max_t = max_t.to_f32().unwrap();
        let (bg_luma, bg_a) = (self.0[0], self.0[1]);
        let (fg_luma, fg_a) = (other.0[0], other.0[1]);

        let (bg_luma, bg_a) = (
            bg_luma.to_f32().unwrap() / max_t,
            bg_a.to_f32().unwrap() / max_t,
        );
        let (fg_luma, fg_a) = (
            fg_luma.to_f32().unwrap() / max_t,
            fg_a.to_f32().unwrap() / max_t,
        );

        let alpha_final = bg_a + fg_a - bg_a * fg_a;
        if alpha_final == 0.0 {
            return;
        };
        let bg_luma_a = bg_luma * bg_a;
        let fg_luma_a = fg_luma * fg_a;

        let out_luma_a = fg_luma_a + bg_luma_a * (1.0 - fg_a);
        let out_luma = out_luma_a / alpha_final;

        *self = $luma_a([
            NumCast::from(max_t * out_luma).unwrap(),
            NumCast::from(max_t * alpha_final).unwrap(),
        ])
    }
}

impl<T: Primitive> Blend for $rgb<T> {
    fn blend(&mut self, other: &$rgb<T>) {
        *self = *other
    }
}

impl<T: Primitive> Blend for $rgba<T> {
    fn blend(&mut self, other: &$rgba<T>) {
        // http://stackoverflow.com/questions/7438263/alpha-compositing-algorithm-blend-modes#answer-11163848

        // First, as we don't know what type our pixel is, we have to convert to floats between 0.0 and 1.0
        let max_t = T::max_value();
        let max_t = max_t.to_f32().unwrap();
        let (bg_r, bg_g, bg_b, bg_a) = (self.0[0], self.0[1], self.0[2], self.0[3]);
        let (fg_r, fg_g, fg_b, fg_a) = (other.0[0], other.0[1], other.0[2], other.0[3]);
        let (bg_r, bg_g, bg_b, bg_a) = (
            bg_r.to_f32().unwrap() / max_t,
            bg_g.to_f32().unwrap() / max_t,
            bg_b.to_f32().unwrap() / max_t,
            bg_a.to_f32().unwrap() / max_t,
        );
        let (fg_r, fg_g, fg_b, fg_a) = (
            fg_r.to_f32().unwrap() / max_t,
            fg_g.to_f32().unwrap() / max_t,
            fg_b.to_f32().unwrap() / max_t,
            fg_a.to_f32().unwrap() / max_t,
        );

        // Work out what the final alpha level will be
        let alpha_final = bg_a + fg_a - bg_a * fg_a;
        if alpha_final == 0.0 {
            return;
        };

        // We premultiply our channels by their alpha, as this makes it easier to calculate
        let (bg_r_a, bg_g_a, bg_b_a) = (bg_r * bg_a, bg_g * bg_a, bg_b * bg_a);
        let (fg_r_a, fg_g_a, fg_b_a) = (fg_r * fg_a, fg_g * fg_a, fg_b * fg_a);

        // Standard formula for src-over alpha compositing
        let (out_r_a, out_g_a, out_b_a) = (
            fg_r_a + bg_r_a * (1.0 - fg_a),
            fg_g_a + bg_g_a * (1.0 - fg_a),
            fg_b_a + bg_b_a * (1.0 - fg_a),
        );

        // Unmultiply the channels by our resultant alpha channel
        let (out_r, out_g, out_b) = (
            out_r_a / alpha_final,
            out_g_a / alpha_final,
            out_b_a / alpha_final,
        );

        // Cast back to our initial type on return
        *self = $rgba([
            NumCast::from(max_t * out_r).unwrap(),
            NumCast::from(max_t * out_g).unwrap(),
            NumCast::from(max_t * out_b).unwrap(),
            NumCast::from(max_t * alpha_final).unwrap(),
        ])
    }
}

    }
}

blend_bit_depth_common!(Luma, LumaA, Rgb, Rgba);
blend_bit_depth_common!(Luma16, LumaA16, Rgb16, Rgba16);

impl<T: Primitive> Blend for Bgra<T> {
    fn blend(&mut self, other: &Bgra<T>) {
        // http://stackoverflow.com/questions/7438263/alpha-compositing-algorithm-blend-modes#answer-11163848

        // First, as we don't know what type our pixel is, we have to convert to floats between 0.0 and 1.0
        let max_t = T::max_value();
        let max_t = max_t.to_f32().unwrap();
        let (bg_r, bg_g, bg_b, bg_a) = (self.0[2], self.0[1], self.0[0], self.0[3]);
        let (fg_r, fg_g, fg_b, fg_a) = (other.0[2], other.0[1], other.0[0], other.0[3]);
        let (bg_r, bg_g, bg_b, bg_a) = (
            bg_r.to_f32().unwrap() / max_t,
            bg_g.to_f32().unwrap() / max_t,
            bg_b.to_f32().unwrap() / max_t,
            bg_a.to_f32().unwrap() / max_t,
        );
        let (fg_r, fg_g, fg_b, fg_a) = (
            fg_r.to_f32().unwrap() / max_t,
            fg_g.to_f32().unwrap() / max_t,
            fg_b.to_f32().unwrap() / max_t,
            fg_a.to_f32().unwrap() / max_t,
        );

        // Work out what the final alpha level will be
        let alpha_final = bg_a + fg_a - bg_a * fg_a;
        if alpha_final == 0.0 {
            return;
        };

        // We premultiply our channels by their alpha, as this makes it easier to calculate
        let (bg_r_a, bg_g_a, bg_b_a) = (bg_r * bg_a, bg_g * bg_a, bg_b * bg_a);
        let (fg_r_a, fg_g_a, fg_b_a) = (fg_r * fg_a, fg_g * fg_a, fg_b * fg_a);

        // Standard formula for src-over alpha compositing
        let (out_r_a, out_g_a, out_b_a) = (
            fg_r_a + bg_r_a * (1.0 - fg_a),
            fg_g_a + bg_g_a * (1.0 - fg_a),
            fg_b_a + bg_b_a * (1.0 - fg_a),
        );

        // Unmultiply the channels by our resultant alpha channel
        let (out_r, out_g, out_b) = (
            out_r_a / alpha_final,
            out_g_a / alpha_final,
            out_b_a / alpha_final,
        );

        // Cast back to our initial type on return
        *self = Bgra([
            NumCast::from(max_t * out_b).unwrap(),
            NumCast::from(max_t * out_g).unwrap(),
            NumCast::from(max_t * out_r).unwrap(),
            NumCast::from(max_t * alpha_final).unwrap(),
        ])
    }
}

impl<T: Primitive> Blend for Bgr<T> {
    fn blend(&mut self, other: &Bgr<T>) {
        *self = *other
    }
}


/// Invert a color
pub(crate) trait Invert {
    /// Inverts a color in-place.
    fn invert(&mut self);
}

macro_rules! invert_bit_depth_common {
    ($luma:ident, $luma_a:ident, $rgb:ident, $rgba:ident) => {

        impl<T: Primitive> Invert for $luma<T> {
            fn invert(&mut self) {
                let l = self.0;

                let max = T::max_value();
                let l1 = max - l[0];

                *self = $luma([l1])
            }
        }

        impl<T: Primitive> Invert for $luma_a<T> {
            fn invert(&mut self) {
                let l = self.0;
                let max = T::max_value();

                *self = $luma_a([max - l[0], l[1]])
            }
        }

        impl<T: Primitive> Invert for $rgb<T> {
            fn invert(&mut self) {
                let rgb = self.0;

                let max = T::max_value();

                let r1 = max - rgb[0];
                let g1 = max - rgb[1];
                let b1 = max - rgb[2];

                *self = $rgb([r1, g1, b1])
            }
        }

        impl<T: Primitive> Invert for $rgba<T> {
            fn invert(&mut self) {
                let rgba = self.0;

                let max = T::max_value();

                *self = $rgba([max - rgba[0], max - rgba[1], max - rgba[2], rgba[3]])
            }
        }
    };
}

invert_bit_depth_common!(Luma, LumaA, Rgb, Rgba);
invert_bit_depth_common!(Luma16, LumaA16, Rgb16, Rgba16);

impl<T: Primitive> Invert for Bgra<T> {
    fn invert(&mut self) {
        let bgra = self.0;

        let max = T::max_value();

        *self = Bgra([max - bgra[2], max - bgra[1], max - bgra[0], bgra[3]])
    }
}

impl<T: Primitive> Invert for Bgr<T> {
    fn invert(&mut self) {
        let bgr = self.0;

        let max = T::max_value();

        let r1 = max - bgr[2];
        let g1 = max - bgr[1];
        let b1 = max - bgr[0];

        *self = Bgr([b1, g1, r1])
    }
}

#[cfg(test)]
mod tests {
    use super::{LumaA, Pixel, Rgb, Rgba, Bgr, Bgra};

    #[test]
    fn test_apply_with_alpha_rgba() {
        let mut rgba = Rgba([0, 0, 0, 0]);
        rgba.apply_with_alpha(|s| s, |_| 0xFF);
        assert_eq!(rgba, Rgba([0, 0, 0, 0xFF]));
    }

    #[test]
    fn test_apply_with_alpha_bgra() {
        let mut bgra = Bgra([0, 0, 0, 0]);
        bgra.apply_with_alpha(|s| s, |_| 0xFF);
        assert_eq!(bgra, Bgra([0, 0, 0, 0xFF]));
    }

    #[test]
    fn test_apply_with_alpha_rgb() {
        let mut rgb = Rgb([0, 0, 0]);
        rgb.apply_with_alpha(|s| s, |_| panic!("bug"));
        assert_eq!(rgb, Rgb([0, 0, 0]));
    }

    #[test]
    fn test_apply_with_alpha_bgr() {
        let mut bgr = Bgr([0, 0, 0]);
        bgr.apply_with_alpha(|s| s, |_| panic!("bug"));
        assert_eq!(bgr, Bgr([0, 0, 0]));
    }


    #[test]
    fn test_map_with_alpha_rgba() {
        let rgba = Rgba([0, 0, 0, 0]).map_with_alpha(|s| s, |_| 0xFF);
        assert_eq!(rgba, Rgba([0, 0, 0, 0xFF]));
    }

    #[test]
    fn test_map_with_alpha_rgb() {
        let rgb = Rgb([0, 0, 0]).map_with_alpha(|s| s, |_| panic!("bug"));
        assert_eq!(rgb, Rgb([0, 0, 0]));
    }

    #[test]
    fn test_map_with_alpha_bgr() {
        let bgr = Bgr([0, 0, 0]).map_with_alpha(|s| s, |_| panic!("bug"));
        assert_eq!(bgr, Bgr([0, 0, 0]));
    }


    #[test]
    fn test_map_with_alpha_bgra() {
        let bgra = Bgra([0, 0, 0, 0]).map_with_alpha(|s| s, |_| 0xFF);
        assert_eq!(bgra, Bgra([0, 0, 0, 0xFF]));
    }

    #[test]
    fn test_blend_luma_alpha() {
        let ref mut a = LumaA([255 as u8, 255]);
        let b = LumaA([255 as u8, 255]);
        a.blend(&b);
        assert_eq!(a.0[0], 255);
        assert_eq!(a.0[1], 255);

        let ref mut a = LumaA([255 as u8, 0]);
        let b = LumaA([255 as u8, 255]);
        a.blend(&b);
        assert_eq!(a.0[0], 255);
        assert_eq!(a.0[1], 255);

        let ref mut a = LumaA([255 as u8, 255]);
        let b = LumaA([255 as u8, 0]);
        a.blend(&b);
        assert_eq!(a.0[0], 255);
        assert_eq!(a.0[1], 255);

        let ref mut a = LumaA([255 as u8, 0]);
        let b = LumaA([255 as u8, 0]);
        a.blend(&b);
        assert_eq!(a.0[0], 255);
        assert_eq!(a.0[1], 0);
    }

    #[test]
    fn test_blend_rgba() {
        let ref mut a = Rgba([255 as u8, 255, 255, 255]);
        let b = Rgba([255 as u8, 255, 255, 255]);
        a.blend(&b);
        assert_eq!(a.0, [255, 255, 255, 255]);

        let ref mut a = Rgba([255 as u8, 255, 255, 0]);
        let b = Rgba([255 as u8, 255, 255, 255]);
        a.blend(&b);
        assert_eq!(a.0, [255, 255, 255, 255]);

        let ref mut a = Rgba([255 as u8, 255, 255, 255]);
        let b = Rgba([255 as u8, 255, 255, 0]);
        a.blend(&b);
        assert_eq!(a.0, [255, 255, 255, 255]);

        let ref mut a = Rgba([255 as u8, 255, 255, 0]);
        let b = Rgba([255 as u8, 255, 255, 0]);
        a.blend(&b);
        assert_eq!(a.0, [255, 255, 255, 0]);
    }    

    #[test]
    fn test_apply_without_alpha_rgba() {
        let mut rgba = Rgba([0, 0, 0, 0]);
        rgba.apply_without_alpha(|s| s + 1);
        assert_eq!(rgba, Rgba([1, 1, 1, 0]));
    }

    #[test]
    fn test_apply_without_alpha_bgra() {
        let mut bgra = Bgra([0, 0, 0, 0]);
        bgra.apply_without_alpha(|s| s + 1);
        assert_eq!(bgra, Bgra([1, 1, 1, 0]));
    }

    #[test]
    fn test_apply_without_alpha_rgb() {
        let mut rgb = Rgb([0, 0, 0]);
        rgb.apply_without_alpha(|s| s + 1);
        assert_eq!(rgb, Rgb([1, 1, 1]));
    }

    #[test]
    fn test_apply_without_alpha_bgr() {
        let mut bgr = Bgr([0, 0, 0]);
        bgr.apply_without_alpha(|s| s + 1);
        assert_eq!(bgr, Bgr([1, 1, 1]));
    }

    #[test]
    fn test_map_without_alpha_rgba() {
        let rgba = Rgba([0, 0, 0, 0]).map_without_alpha(|s| s + 1);
        assert_eq!(rgba, Rgba([1, 1, 1, 0]));
    }

    #[test]
    fn test_map_without_alpha_rgb() {
        let rgb = Rgb([0, 0, 0]).map_without_alpha(|s| s + 1);
        assert_eq!(rgb, Rgb([1, 1, 1]));
    }

    #[test]
    fn test_map_without_alpha_bgr() {
        let bgr = Bgr([0, 0, 0]).map_without_alpha(|s| s + 1);
        assert_eq!(bgr, Bgr([1, 1, 1]));
    }

    #[test]
    fn test_map_without_alpha_bgra() {
        let bgra = Bgra([0, 0, 0, 0]).map_without_alpha(|s| s + 1);
        assert_eq!(bgra, Bgra([1, 1, 1, 0]));
    }
}
