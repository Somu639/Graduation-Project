---
name: Deep Space Research Lab
colors:
  surface: '#131313'
  surface-dim: '#131313'
  surface-bright: '#393939'
  surface-container-lowest: '#0e0e0e'
  surface-container-low: '#1c1b1b'
  surface-container: '#201f1f'
  surface-container-high: '#2a2a2a'
  surface-container-highest: '#353534'
  on-surface: '#e5e2e1'
  on-surface-variant: '#bccbb9'
  inverse-surface: '#e5e2e1'
  inverse-on-surface: '#313030'
  outline: '#869585'
  outline-variant: '#3d4a3d'
  surface-tint: '#53e076'
  primary: '#53e076'
  on-primary: '#003914'
  primary-container: '#1db954'
  on-primary-container: '#004118'
  inverse-primary: '#006e2d'
  secondary: '#3de96f'
  on-secondary: '#003913'
  secondary-container: '#00cc57'
  on-secondary-container: '#004f1d'
  tertiary: '#c8c6c5'
  on-tertiary: '#303030'
  tertiary-container: '#a2a1a1'
  on-tertiary-container: '#383838'
  error: '#ffb4ab'
  on-error: '#690005'
  error-container: '#93000a'
  on-error-container: '#ffdad6'
  primary-fixed: '#72fe8f'
  primary-fixed-dim: '#53e076'
  on-primary-fixed: '#002108'
  on-primary-fixed-variant: '#005320'
  secondary-fixed: '#69ff89'
  secondary-fixed-dim: '#34e36a'
  on-secondary-fixed: '#002108'
  on-secondary-fixed-variant: '#00531f'
  tertiary-fixed: '#e4e2e1'
  tertiary-fixed-dim: '#c8c6c5'
  on-tertiary-fixed: '#1b1c1c'
  on-tertiary-fixed-variant: '#474746'
  background: '#131313'
  on-background: '#e5e2e1'
  surface-variant: '#353534'
typography:
  display-lg:
    fontFamily: Inter
    fontSize: 48px
    fontWeight: '700'
    lineHeight: 56px
    letterSpacing: -0.02em
  headline-lg:
    fontFamily: Inter
    fontSize: 32px
    fontWeight: '700'
    lineHeight: 40px
    letterSpacing: -0.01em
  headline-md:
    fontFamily: Inter
    fontSize: 24px
    fontWeight: '600'
    lineHeight: 32px
  headline-sm:
    fontFamily: Inter
    fontSize: 18px
    fontWeight: '600'
    lineHeight: 24px
  body-lg:
    fontFamily: Inter
    fontSize: 16px
    fontWeight: '400'
    lineHeight: 24px
  body-md:
    fontFamily: Inter
    fontSize: 14px
    fontWeight: '400'
    lineHeight: 20px
  label-md:
    fontFamily: Inter
    fontSize: 12px
    fontWeight: '600'
    lineHeight: 16px
    letterSpacing: 0.05em
  label-sm:
    fontFamily: Inter
    fontSize: 11px
    fontWeight: '500'
    lineHeight: 14px
rounded:
  sm: 0.25rem
  DEFAULT: 0.5rem
  md: 0.75rem
  lg: 1rem
  xl: 1.5rem
  full: 9999px
spacing:
  container-margin: 32px
  gutter: 24px
  card-padding: 24px
  stack-sm: 8px
  stack-md: 16px
  stack-lg: 32px
---

## Brand & Style

This design system draws inspiration from high-fidelity audio interfaces to create a focused, immersive environment for product management and UX research. The aesthetic is rooted in **Modern Minimalism** with a distinct **Dark Mode** foundation. By using deep charcoal surfaces and high-contrast accents, the interface reduces visual noise, allowing data visualizations and research insights to command full attention.

The target audience consists of product owners and researchers who require long-duration focus. The emotional response is one of calm authority and precision. The style utilizes subtle depth through tonal layering rather than heavy shadows, maintaining a "glass-on-ebony" feel that is professional, sleek, and highly functional.

## Colors

The palette is strictly dark-themed to minimize eye strain and maximize the "pop" of data points. 

- **Primary & Secondary:** The iconic "Spotify Green" is reserved for high-priority actions and active states. The hover state shifts to a more luminous green to provide immediate tactile feedback.
- **Surface Hierarchy:** The background uses a true-dark base (#121212). Secondary panels and navigation use a slightly elevated grey (#181818), while interactive or floating cards use the highest tonal elevation (#202020).
- **Borders:** All borders must be "hairline" (1px or 0.5px where supported) using #2A2A2A to define structure without creating visual clutter.

## Typography

The design system utilizes **Inter** exclusively to lean into a systematic, utilitarian aesthetic. 

- **Hierarchy:** Bold weights are used for headlines to create clear entry points. Muted text (#B3B3B3) is applied to body-md and labels to establish secondary hierarchy.
- **Readability:** For data-dense dashboards, use `body-md` (14px) as the standard for tabular data and research notes.
- **Styling:** Labels should often use uppercase with slight letter spacing to differentiate metadata from content.

## Layout & Spacing

This design system uses a **Fluid Grid** model with generous margins to prevent the dashboard from feeling claustrophobic despite the data density.

- **Grid:** A 12-column grid is standard for desktop. On tablet, this shifts to an 8-column layout. 
- **Rhythm:** An 8px linear scale governs all spacing.
- **Density:** While the design is "uncluttered," horizontal bar charts and heatmaps should utilize the full width of their parent containers to maximize data resolution.
- **Mobile:** Elements reflow into a single column with a 16px side margin. Headlines scale down (e.g., `headline-lg` moves to 24px) to fit smaller viewports.

## Elevation & Depth

Visual hierarchy is achieved through **Tonal Layers** and **Hairline Outlines**.

- **Elevation 0 (Base):** #121212. Used for the main background.
- **Elevation 1 (Panels):** #181818. Used for sidebars, top bars, and secondary layout containers.
- **Elevation 2 (Cards):** #202020. Used for interactive cards and modals.
- **Shadows:** Avoid drop shadows for standard UI elements. Use a very soft, large-radius black shadow (30% opacity) only for floating modals or context menus to subtly separate them from the grid.

## Shapes

The shape language is a mix of structured containers and organic interactive elements.

- **Containers:** Dashboard cards and panels use a consistent **12px (0.75rem)** corner radius (`rounded-lg` in this system).
- **Interactive Elements:** Buttons and tags use a **Pill** shape (fully rounded) to contrast against the rectangular grid of the dashboard.
- **Visual Accents:** Research quote cards feature a 4px solid vertical green border on the left edge to denote "Insight" status.

## Components

- **Primary Buttons:** Pill-shaped. Background: #1DB954; Text: #121212 (Bold). On hover, background shifts to #1ED760.
- **Metrics Cards:** Use `surface-card` (#202020) with a hairline border. Large `headline-lg` for the value and `label-md` (Muted) for the description.
- **Horizontal Tab Bar:** Text-only labels with #B3B3B3 color. The active state features #FFFFFF text and a 2px #1DB954 underline or a small green dot indicator below the text.
- **Input Fields:** Background: #2A2A2A; Border: 1px transparent. On focus, the border becomes #1DB954.
- **Severity Heatmaps:** Use a monochromatic green scale for research impact, where the darkest green (#1DB954) represents the highest severity, and #2A2A2A represents null/low.
- **Quote Cards:** Elevated surface (#202020) with a 4px #1DB954 left-accent border. Text should be italicized `body-lg`.