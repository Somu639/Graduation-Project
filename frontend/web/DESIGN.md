---
name: Luminous Discovery
colors:
  surface: '#f8f9fa'
  surface-dim: '#d9dadb'
  surface-bright: '#f8f9fa'
  surface-container-lowest: '#ffffff'
  surface-container-low: '#f3f4f5'
  surface-container: '#edeeef'
  surface-container-high: '#e7e8e9'
  surface-container-highest: '#e1e3e4'
  on-surface: '#191c1d'
  on-surface-variant: '#3d4a3d'
  inverse-surface: '#2e3132'
  inverse-on-surface: '#f0f1f2'
  outline: '#6d7b6c'
  outline-variant: '#bccbb9'
  surface-tint: '#006e2d'
  primary: '#006e2d'
  on-primary: '#ffffff'
  primary-container: '#1db954'
  on-primary-container: '#004118'
  inverse-primary: '#53e076'
  secondary: '#645d5c'
  on-secondary: '#ffffff'
  secondary-container: '#e8dddd'
  on-secondary-container: '#696161'
  tertiary: '#5d5f5f'
  on-tertiary: '#ffffff'
  tertiary-container: '#a0a1a1'
  on-tertiary-container: '#363838'
  error: '#ba1a1a'
  on-error: '#ffffff'
  error-container: '#ffdad6'
  on-error-container: '#93000a'
  primary-fixed: '#72fe8f'
  primary-fixed-dim: '#53e076'
  on-primary-fixed: '#002108'
  on-primary-fixed-variant: '#005320'
  secondary-fixed: '#ebe0df'
  secondary-fixed-dim: '#cfc4c4'
  on-secondary-fixed: '#201a1a'
  on-secondary-fixed-variant: '#4c4545'
  tertiary-fixed: '#e2e2e2'
  tertiary-fixed-dim: '#c6c6c7'
  on-tertiary-fixed: '#1a1c1c'
  on-tertiary-fixed-variant: '#454747'
  background: '#f8f9fa'
  on-background: '#191c1d'
  surface-variant: '#e1e3e4'
typography:
  display-lg:
    fontFamily: Plus Jakarta Sans
    fontSize: 48px
    fontWeight: '800'
    lineHeight: 56px
    letterSpacing: -0.02em
  headline-lg:
    fontFamily: Plus Jakarta Sans
    fontSize: 32px
    fontWeight: '700'
    lineHeight: 40px
    letterSpacing: -0.01em
  headline-lg-mobile:
    fontFamily: Plus Jakarta Sans
    fontSize: 24px
    fontWeight: '700'
    lineHeight: 32px
  headline-md:
    fontFamily: Plus Jakarta Sans
    fontSize: 20px
    fontWeight: '700'
    lineHeight: 28px
  body-lg:
    fontFamily: Plus Jakarta Sans
    fontSize: 16px
    fontWeight: '500'
    lineHeight: 24px
  body-md:
    fontFamily: Plus Jakarta Sans
    fontSize: 14px
    fontWeight: '400'
    lineHeight: 20px
  label-md:
    fontFamily: Hanken Grotesk
    fontSize: 12px
    fontWeight: '600'
    lineHeight: 16px
    letterSpacing: 0.05em
rounded:
  sm: 0.25rem
  DEFAULT: 0.5rem
  md: 0.75rem
  lg: 1rem
  xl: 1.5rem
  full: 9999px
spacing:
  unit: 8px
  container-padding-mobile: 16px
  container-padding-desktop: 32px
  gutter: 24px
  card-gap: 16px
---

## Brand & Style

This design system reimagines a world-leading music identity through a high-clarity, light-mode lens. The brand personality is **intelligent, personalized, and energetic**, shifting away from the dark-room "studio" aesthetic toward an "airy gallery" feel. It targets a modern audience that values data-driven discovery and seamless navigation.

The design style utilizes **Modern Minimalism** with **Tactile accents**. It leverages heavy whitespace to let album art and metadata breathe, while using the signature primary green to signal interaction and energy. The emotional response is one of clarity and optimism—making the vast landscape of music feel approachable and curated.

## Colors

The palette is anchored by the iconic **Spotify Green**, used exclusively for primary actions, playback progress, and active states. The background uses a crisp **Soft White** (#F8F9FA) to ensure the interface feels lightweight and fresh. 

**Secondary Black** (#191414) is reserved for high-contrast typography to ensure WCAG AAA readability. **Light Gray** (#E9ECEF) serves as a subtle structural color for borders and secondary container backgrounds, preventing the interface from feeling "washed out" while maintaining a soft, approachable depth.

## Typography

The typography system relies on **Plus Jakarta Sans** for its friendly, geometric, and modern construction, mimicking the clean lines of premium circular fonts. Headlines are set with tight tracking and heavy weights to create a "bold" editorial hierarchy.

For technical metadata and secondary labels (like timestamps or genre tags), **Hanken Grotesk** is used to provide a precise, high-legibility contrast. The scale ensures that even on dense discovery dashboards, the information hierarchy is immediate. Mobile headers are scaled down to maintain screen real estate while retaining their bold visual weight.

## Layout & Spacing

The layout utilizes a **12-column fluid grid** for desktop and a **4-column grid** for mobile devices. A strict 8px spacing power-of-two scale is enforced to ensure rhythmic consistency.

- **Margins:** 32px on desktop to provide a "breathing room" feel; 16px on mobile for maximum content density.
- **Gutters:** 24px fixed gutters to separate music cards and navigation panels.
- **Sectioning:** Content is grouped into horizontal scrolling "shelves" for discovery, with vertical stacks reserved for primary library navigation.

## Elevation & Depth

This design system eschews heavy shadows in favor of **Tonal Layering** and **Subtle Ambient Depth**. 

- **Level 0 (Background):** Soft White (#F8F9FA).
- **Level 1 (Cards/Sidebar):** Pure White (#FFFFFF) with a 1px border (#E9ECEF).
- **Level 2 (Hover/Active):** A very soft, diffused 12% opacity shadow with a 16px blur, creating a "lifted" effect without appearing muddy.
- **Interactive Elements:** Buttons use high-contrast fills. Secondary inputs use a subtle inset shadow to indicate "tappable" regions.

## Shapes

The shape language is defined by **pronounced, friendly curves**. Base components like buttons and inputs use a **0.5rem (8px)** radius. Larger structural elements, such as album art containers and dashboard cards, use a **1rem (16px)** radius to reinforce the modern, approachable vibe. 

Circular shapes are reserved exclusively for:
- User avatars.
- Play/Pause floating action buttons.
- Artist profile images.

## Components

### Buttons
- **Primary:** Spotify Green background, White text. Bold weight. Rounded-pill shape.
- **Secondary:** Transparent background, 1px Gray border, Black text. 
- **Ghost:** No background or border, Green text for high-importance links, Black for low-importance.

### Cards
Discovery cards use a White background with 16px rounded corners. Album art within the card should have a slightly smaller 12px radius to create a nested, organic feel. High-contrast typography for the "Song Title" and Gray text for the "Artist Name."

### Playback Controls
The progress bar uses a light gray track with a Spotify Green fill. The "scrubber" thumb is only visible on hover to maintain a clean aesthetic.

### Chips & Tags
Used for AI-generated genres (e.g., "Mellow," "Focus"). These should have a light gray background (#E9ECEF), no border, and use the Label-MD typography style.

### Input Fields
Search bars should be 100% width, using the Soft White background with an 8px radius. The placeholder text should be a mid-tone gray to ensure the UI feels light.