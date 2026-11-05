interface UploadedFilePreviewImageProps {
  src: string | null;
  alt: string;
  className: string;
}

// Renders a local file-upload preview from a blob: URL (URL.createObjectURL on a
// File the user just picked). Kept in its own file so this one img src can be
// excluded from js/xss-through-dom scanning (see codeql-config.yml) without
// losing CodeQL coverage on the rest of CompareUI.tsx -- see that exclude's
// comment for why CodeQL flags this as a sink despite it never being attacker-
// controlled or executing as HTML.
function UploadedFilePreviewImage({ src, alt, className }: UploadedFilePreviewImageProps) {
  return <img src={src || ""} alt={alt} className={className} />;
}

export default UploadedFilePreviewImage;
