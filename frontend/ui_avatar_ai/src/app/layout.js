import "../styles/globals.css";

export const metadata = {
    title: "3D Avatar Generator",
    description: "Upload images to generate a realistic 3D human avatar.",
};

export default function RootLayout({ children }) {
    return (
        <html lang="en">
            <body className="bg-black text-white">{children}</body>
        </html>
    );
}
