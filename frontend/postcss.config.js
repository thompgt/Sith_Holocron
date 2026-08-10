// Tailwind v4 moved its PostCSS plugin into a separate package; listing bare
// `tailwindcss` here (the v3 form) makes the build fail outright. v4 also does
// its own vendor prefixing via Lightning CSS, so autoprefixer is no longer
// needed in this pipeline.
export default {
  plugins: {
    '@tailwindcss/postcss': {},
  },
}
