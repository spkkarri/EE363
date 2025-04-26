import Latex from 'react-latex-next';
import 'katex/dist/katex.min.css'; // Ensure KaTeX styles are applied
import './FormulaWidget.css';

function FormulaWidget({ formula, imageUrl }) {
  return (
    <div className="formula-widget">
      {formula && (
        <div className="formula-container">
          <Latex>{formula}</Latex>
        </div>
      )}
      {imageUrl && (
        <div className="image-container">
          <img src={imageUrl} alt="Formula-related image" className="formula-image" />
        </div>
      )}
    </div>
  );
}

export default FormulaWidget;
