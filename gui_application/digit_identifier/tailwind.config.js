/** @type {import('tailwindcss').Config} */
module.exports = {
    content: ['./src/**/*.{js,jsx}'],
    theme: {
        extend: {},
        container: {
            center: true,
            padding: {
                DEFAULT: '.5rem',
                sm: '2rem',
            },
        },
    },
    plugins: [],
};
