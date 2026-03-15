import React from 'react';
import Footer from '@theme-original/DocItem/Footer';
import EditThisPage from '@site/src/components/EditThisPage';

export default function FooterWrapper(props) {
  return (
    <>
      <Footer {...props} />
      <EditThisPage />
    </>
  );
} 